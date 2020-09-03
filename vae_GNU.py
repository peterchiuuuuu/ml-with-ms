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


	###vae
	latent_dim = 156
	input_ = Input(shape = (34,22), name = 'input')
	'''
	x = Conv1D(2, 5, activation='tanh', padding='same', name = 'e1')(input_)
	x = Conv1D(2, 5, activation='tanh', padding='same')(x)
	x = Conv1D(1, 4, activation='tanh', padding='same')(x)
	'''
	x = Conv1D(16, 8, activation='tanh', name = 'e1')(input_)
	x = BatchNormalization(axis=-1,name='encoder_dense1_norm')(x)
	x = Conv1D(32, 10, activation='tanh')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense2_norm')(x)
	x = Conv1D(64, 12, activation='tanh')(x)
	x = BatchNormalization(axis=-1,name='encoder_dense3_norm')(x)
	#x = Dense(latent_dim, activation  = 'tanh')(x)

	x = Flatten()(x)

	##hidden layers
	z_mean = Dense(latent_dim)(x)
	z_log_var = Dense(latent_dim)(x)

	#z_mean_log_var_output = Concatenate(name='z_mean_log_var')([z_mean, z_log_var])

	z = Lambda(sampling, name='z')([z_mean, z_log_var])
	encoder = Model(input_, [z_mean, z_log_var, z])
	#encoder.summary()
	input_d  = Input(shape=(latent_dim,))

	x_dec = Dense(100)(input_d)
	x_dec = BatchNormalization(axis=-1,name='decoder_dense_norm')(x_dec)
	###

	x_dec = RepeatVector(34)(x_dec)
	'''
	dd = Conv1D(10, 5, activation='tanh', padding='same')(dd)
	dd = Conv1D(22, 5, activation='softmax', padding='same')(dd)
	'''
	#x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru0")(x_dec)
	x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru1")(x_dec)
	x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru2")(x_dec)
	#x_dec = GRU(50, return_sequences=True, activation='tanh', name="decoder_gru3")(x_dec)
	x_dec = GRU(22, return_sequences=True, activation='softmax', name='decoder_gru_final')(x_dec)

	#outputs = decoder(encoder(inputs)[2])
	#x = GRU(500)(x)
	#x = GRU(500)(x)
	#decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
	decoder = Model(input_d, x_dec)
	                
	                
	vae_outputs = decoder(encoder(input_)[2])
	vae         = Model(input_, vae_outputs, name='vae')
	
	####
	#data collecting
	
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
	    #reconstruction_loss = categorical_crossentropy(K.flatten(true), K.flatten(pred))
	    reconstruction_loss *= 34*22
	    #* img_width * img_height
	    # KL divergence loss
	    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	    #kl_loss = K.sum(kl_loss, axis=-1)
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

	####save every epocch
	filepath="weights-improvement-{epoch:02d}.hdf5"
	#checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	checkpoint = ModelCheckpoint(filepath, monitor='kl_reconstruction_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	###
	opt = keras.optimizers.Adam(lr=0.0003)

	#train
	vae.compile(optimizer=opt, loss = kl_reconstruction_loss, metrics=[kl_loss, reconstruct_error])
	#autoencoder.fit(train_images, train_images)
	'''
	vae.fit(x_train, x_train,
	                epochs=120,
	                batch_size=250,
	                shuffle=True)
	                #validation_data=(x_test, x_test))
	                '''
	vae.fit(x_train, x_train,
	                epochs=20,
	                batch_size=250,
	                shuffle=True,
	                callbacks=callbacks_list) 

	#####save model
	vae.save("propertie_model")
	encoder.save("propertie_model_encoder")
	decoder.save("propertie_model_decoder")
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
	#print(liss)


	print('total time:' + str(time.time() - start_time) + 'sec')
if __name__ == '__main__':
	main()