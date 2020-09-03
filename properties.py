import tensorflow as tf
from keras.models import Model
import keras

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
#import PIL
#import imageio
#import pymol
import csv
#sys.path.insert(0, os.path.abspath('/Users/peterchiu/miniconda3/envs/my-rdkit-env/bin/'))
#sys.path.append("/Users/peterchiu/miniconda3/envs/my-rdkit-env/bin/")
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import random

import pandas as pd
from IPython import display
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, UpSampling2D, UpSampling1D, MaxPooling1D, Lambda
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Flatten, RepeatVector, Dropout
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from rdkit import RDLogger  
import time
RDLogger.DisableLog('rdApp.*') 



def add_space(raw_data, input_dim = 34):
    out = []
    for i in raw_data:
        if len(i) < input_dim:
            out.append(i+' '*(input_dim - len(i)))
        else:
            out.append(i)           
    return(out)

def plot_auto(out, predict_st):
    size = (50, 50)
    #print(out)
    fig = Draw.MolToMPL(Chem.MolFromSmiles(out), size=size)
    #print(predict_st)
    fig = Draw.MolToMPL(Chem.MolFromSmiles(predict_st), size=size)

def sampling(args):
        z_mean, z_log_var = args
        
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        #epsilon = K.random_normal_variable(shape=(batch,dim),mean=0., scale=1.)
        # insert kl loss here

        z_rand = z_mean + K.exp(z_log_var / 2) * epsilon
        return K.in_train_phase(z_rand, z_mean)


def main():
	####
	start_time = time.time()

	###
	data_size = 25000



	####
	SMILES_CHARS = [' ', 'C', 'N', 'O', '#', '=', '1', '(', ')', 'c', '[', 'n', 'H',']', 'o',
	 '3', '+','-', '2', 'F', '4', '5']
	def smiles_encoder(smiles, maxlen=34):
		#print(smiles)
		#smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
		#print(smiles)
		X = np.zeros((maxlen, 22))
		for i, c in enumerate(smiles):
		    #print(i)
		    #print(c)
		    X[i, smi2index[c]] = 1
		return X
	 
	def smiles_decoder( X ):
	    smi = ''
	    X = X.argmax( axis=-1 )
	    for i in X:
	        smi += index2smi[ i ]
	    return smi
	smi2index = dict( (c,i) for i,c in enumerate(SMILES_CHARS))
	index2smi = dict( (i,c) for i,c in enumerate(SMILES_CHARS))

	out_raw = pd.read_csv('QM9_smiles.csv', header = None)[0]
	out = add_space(out_raw)




	QM9_hot = []
	for i in out:
	    #print(i)
	    QM9_hot.append(smiles_encoder(i))

	#read properties data
	QM9_properties = pd.read_csv('QM9_properties.csv') 
	QM9_properties.columns = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
	'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']


	###vae
	latent_dim = 156
	input_ = Input(shape = (34,22), name = 'input')
	'''
	x = Conv1D(2, 5, activation='tanh', padding='same', name = 'e1')(input_)
	x = BatchNormalization(axis=-1,name='encoder_dense1_norm')(x)
	x = Conv1D(2, 5, activation='tanh', padding='same')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense2_norm')(x)
	x = Conv1D(1, 4, activation='tanh', padding='same')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense3_norm')(x)
	
	x = Conv1D(16, 8, activation='tanh', name = 'e1')(input_)
	x = BatchNormalization(axis=-1,name='encoder_dense1_norm')(x)
	x = Conv1D(32, 10, activation='tanh')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense2_norm')(x)
	x = Conv1D(64, 12, activation='tanh')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense3_norm')(x)
	'''
	#x = Dense(latent_dim, activation  = 'tanh')(x)
	x = Conv1D(8, 8, activation='tanh', padding='same', name = 'e1')(input_)
	x = BatchNormalization()(x)
	x = Conv1D(10, 10, activation='tanh', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv1D(12, 12, activation='tanh', padding='same')(x)
	x = BatchNormalization()(x)

	x = Flatten()(x)

	##hidden layers
	z_mean = Dense(latent_dim)(x)
	z_log_var = Dense(latent_dim)(x)

	#z_mean_log_var_output = Concatenate(name='z_mean_log_var')([z_mean, z_log_var])

	z = Lambda(sampling, name='z')([z_mean, z_log_var])
	encoder = Model(input_, [z_mean, z_log_var, z])
	#encoder.summary()
	input_d  = Input(shape=(latent_dim,))

	xp = Dense(67, activation='tanh')(input_d)
	xp = Dropout(0.15)(xp)
	xp = Dense(67, activation='tanh')(xp)
	xp = Dropout(0.15)(xp)
	xp = Dense(67, activation='tanh')(xp)
	xp = Dropout(0.15)(xp)
	'''
	xp = Dense(1000, activation='tanh')(xp)
	xp = Dropout(0.2)(xp)
	xp = Dense(1000, activation='tanh')(xp)
	xp = Dropout(0.2)(xp)
	'''
	xp = Dense(1, activation='linear')(xp)
	properties = Model(input_d, xp)
	                
	                
	properties_outputs = properties(encoder(input_)[2])
	propertie_model	= Model(input_, properties_outputs, name='propertie_model')
	####
	#data collecting
	
	x_train = QM9_hot
	x_train = np.reshape(x_train, (len(x_train), 34, 22))
	y_train = QM9_properties['lumo']
	y_train = np.reshape(y_train, (len(y_train),))
	x_test = QM9_hot[10000:10500]
	x_test = np.reshape(x_test, (len(x_test), 34, 22))



	def kl_mse_loss(true, pred):
		# Reconstruction loss
		mse_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))
		#mse_loss = tf.keras.losses.MSE(true, pred)
		#mse_loss = tf.keras.losses.MeanSquaredError(K.flatten(true), K.flatten(pred))
		#reconstruction_loss = categorical_crossentropy(K.flatten(true), K.flatten(pred))
		#reconstruction_loss *= 34*22
		#* img_width * img_height
		# KL divergence loss
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		#kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		# Total loss = 50% rec + 50% KL divergence loss
		return K.mean(mse_loss + kl_loss)
	def kl_loss(true, pred):
	    kl = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	    kl *= -0.5
	    return(kl)
	def mse_loss(true, pred):
		#mse_loss = tf.keras.losses.MSE(true, pred)
		mse_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))
		return(mse_loss)

	opt = keras.optimizers.Adam(lr=0.0003)

	#train
	propertie_model.compile(optimizer=opt, loss = kl_mse_loss, metrics=[kl_loss, mse_loss])
	#autoencoder.fit(train_images, train_images)
	'''
	vae.fit(x_train, x_train,
	                epochs=120,
	                batch_size=250,
	                shuffle=True)
	                #validation_data=(x_test, x_test))
	                '''
	propertie_model.fit(x_train, y_train,
	                epochs=20,
	                batch_size=250,
	                shuffle=True)
	                #callbacks=callbacks_list) 
	propertie_model.save("propertie_model")
	encoder.save("propertie_model_encoder")
	properties.save("propertie_model_decoder")
	print('total time:' + str(time.time() - start_time) + 'sec')
if __name__ == '__main__':
	main()


