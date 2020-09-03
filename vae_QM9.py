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
from keras.losses import mse, binary_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import backend as K
from rdkit import RDLogger  
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
        #return(z_mean)
        #return(epsilon)

def main():
	###
	data_size = 25000
	batch_size = 250
	epochs = 20
	latent_dim = 156

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


	###vae
	input_ = Input(shape = (34,22), name = 'input')
	x = Conv1D(2, 5, activation='tanh', name = 'e1')(input_)
	x = Conv1D(2, 5, activation='tanh',)(x)
	x = Conv1D(1, 4, activation='tanh',)(x)
	#x = BatchNormalization(axis=-1, name="encoder_norm0")(x)
	#x = Dense(latent_dim, activation  = 'tanh')(x)

	x = Flatten()(x)

	##hidden layers
	z_mean = Dense(latent_dim)(x)
	z_log_var = Dense(latent_dim)(x)


	z = Lambda(sampling, name='z')([z_mean, z_log_var])
	encoder = Model(input_, [z_mean, z_log_var, z])
	

	#####decoder
	input_d  = Input(shape=(latent_dim,))
	#input_d  = Input(shape=(K.int_shape(x)[1],))
	x_dec = RepeatVector(34)(input_d)

	#x = Conv1D(10, 5, activation='tanh', padding='same')(x)
	x_dec = Conv1D(22, 5, activation='softmax', padding='same')(x_dec)
	#x_dec = Dense(22*34, activation  = 'sigmoid')(input_d)
	#x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru0")(x_dec)
	#x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru1")(x_dec)
	#x_dec = GRU(34, return_sequences=True, activation='softmax', name='decoder_gru_final')(x_dec)
	decoder = Model(input_d, x_dec)
	                
	                
	vae_outputs = decoder(encoder(input_)[2])
	vae         = Model(input_, vae_outputs, name='vae')
	
	####
	#data collecting
	#'''
	x_train = QM9_hot
	x_train = np.reshape(x_train, (len(x_train), 34, 22))
	x_test = QM9_hot[10000:10500]
	x_test = np.reshape(x_test, (len(x_test), 34, 22))
	'''
	#random data
	randomlist = random.sample(range(0, 133885), data_size)
	x_train = []
	for i in randomlist:
	    x_train.append(QM9_hot[i])
	#test = QM9_hot[1000:1500]
	x_train = np.reshape(x_train, (len(randomlist), 34, 22))
	'''

	###custom loss function
	def kl_reconstruction_loss(true, pred):
	    # Reconstruction loss
	    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred))
	    reconstruction_loss *= 34*22
	    #* img_width * img_height
	    # KL divergence loss
	    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	    kl_loss = K.sum(kl_loss, axis=-1)
	    kl_loss *= -0.5
	    # Total loss = 50% rec + 50% KL divergence loss
	    return K.mean(reconstruction_loss + kl_loss)
	def kl_loss(true, pred):
	    kl = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	    kl *= -0.5
	    return(kl)
	def reconstruct_error(true, pred):
	    reconstruct_error = binary_crossentropy(K.flatten(true), K.flatten(pred))
	    reconstruct_error *= 34*22
	    return(reconstruct_error)



	#train
	vae.compile(optimizer='adam', loss = kl_reconstruction_loss, metrics=[kl_loss, reconstruct_error])
	#autoencoder.fit(train_images, train_images)
	vae.fit(x_train, x_train,
	                epochs=epochs,
	                batch_size=batch_size,
	                shuffle=True)
	                #validation_data=(x_test, x_test))
	###predict
	predict_raw = vae.predict(x_train)
	predict_st = []
	for i in predict_raw:
	    predict_st.append(smiles_decoder(i))
	###invalid cal
	invalid = 0
	oo = []
	for i, x in enumerate(predict_st):
	    #print(x)
	    m = Chem.MolFromSmiles(x, sanitize=False)
	    if m is None:
	        invalid += 1
	    else:
	        try:
	            Chem.SanitizeMol(m)
	            oo.append(i)
	        except:
	            invalid += 1
	print('invalid num  =' + str(invalid))
	print('invalid rate  =' + str(invalid/len(predict_st)))
	accuracy = 0
	for i in range(len(predict_st)):
		if predict_st[i] == out[i]:
			accuracy += 1
	print('accuracy rate = ' + str(accuracy/len(predict_st)))
 	#print(oo[0])
	#Draw.MolToFile(Chem.MolFromSmiles(out[0]), 'raw.png')
	#Draw.MolToFile(Chem.MolFromSmiles(predict_st[0]), 'pre.png')  
	liss = []
	for i in oo:
		if predict_st[i] not in liss:
			liss.append(predict_st[i])
	print(len(liss))
	print(liss)



if __name__ == '__main__':
	main()