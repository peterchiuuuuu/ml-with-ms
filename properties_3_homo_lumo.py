#######enviroment requirement:
#######tensorflow version: 1.14.0
#######keras version: 2.1.2

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
from sklearn.model_selection import train_test_split
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
def identity(x):
    return K.identity(x)


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
    x = Conv1D(16, 8, activation='tanh', padding='same', name = 'e1')(input_)
    x = BatchNormalization()(x)
    x = Conv1D(32, 10, activation='tanh', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 12, activation='tanh', padding='same')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)


    ##hidden layers
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    z_mean_log_var_output = Concatenate(name='kl_loss')([z_mean, z_log_var])

    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = Model(input_, [z_mean_log_var_output, z])
    encoder_output = [z_mean_log_var_output, z]
    #edecoder
    input_d  = Input(shape=(latent_dim,))

    x_dec = Dense(156)(input_d)
    x_dec = BatchNormalization(axis=-1,name='decoder_dense_norm')(x_dec)
    ###

    x_dec = RepeatVector(34)(x_dec)
    x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru1")(x_dec)
    x_dec = GRU(500, return_sequences=True, activation='tanh', name="decoder_gru2")(x_dec)
    #x_dec = GRU(50, return_sequences=True, activation='tanh', name="decoder_gru3")(x_dec)
    x_dec = GRU(22, return_sequences=True, activation='softmax', name='decoder_gru3')(x_dec)
    decoder = Model(input_d, x_dec)

    #properties
    input_p  = Input(shape=(latent_dim,))
	#xp = Dense(67, activation='tanh')(input_p)
	#xp = Dropout(0.15)(xp)
	#xp = Dense(67, activation='tanh')(xp)
	#xp = Dropout(0.15)(xp)
	#xp = Dense(67, activation='tanh')(xp)
	#xp = Dropout(0.15)(xp)
    xp = Dense(1000, activation='linear')(input_p)
    xp = Dropout(0.2)(xp)
    xp = Dense(1000, activation='linear')(xp)
    xp = Dropout(0.2)(xp)
    xp = Dense(1, activation='linear')(xp)
    pp_homo = Model(input_p, xp) 
    pp_lumo = Model(input_p, xp)                
    #####data aggregate	
    vae_outputs = decoder(encoder(input_)[1])
    vae_outputs = Lambda(identity, name='x_pred')(vae_outputs)               
    homo_outputs = pp_homo(encoder(input_)[1])
    lumo_outputs = pp_lumo(encoder(input_)[1])
    homo_outputs = Lambda(identity, name='homo_loss')(homo_outputs)
    lumo_outputs = Lambda(identity, name='lumo_loss')(lumo_outputs)
    properties_homo	= Model(input_, homo_outputs, name='properties_homo')
    properties_lumo	= Model(input_, lumo_outputs, name='properties_lumo')

    total_ouput = encoder_output
    total_ouput.append(vae_outputs)
    total_ouput.append(homo_outputs)
    total_ouput.append(lumo_outputs)
    total_model = Model(input_, total_ouput)
	####
	#data collecting
	
    x_train = QM9_hot
    x_train = np.reshape(x_train, (len(x_train), 34, 22))
	#y_homo = QM9_properties['homo']
	#y_lumo = QM9_properties['lumo']
	#y_homo = np.reshape(y_homo, (len(y_homo),))
	#y_lumo = np.reshape(y_lumo, (len(y_lumo),))

    x_raw = np.reshape(x_train, (len(x_train), 34, 22))
    y_raw = QM9_properties[['homo', 'lumo']]
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.2, random_state=42)
    #x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5, random_state=42)
    y_homo = np.array(y_train['homo'])
    y_lumo = np.array(y_train['lumo'])
    y_homo_test = np.array(y_test['homo'])
    y_lumo_test = np.array(y_test['lumo'])

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
    	#mse_loss = tf.keras.losses.MeanSquaredError(K.flatten(true), K.flatten(pred), reduction=tf.keras.losses.Reduction.SUM)
    	#mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    	#mse(y_true, y_pred).numpy()
    	mse_loss *= 250

    	return(mse_loss)
    def reconstruct_error(true, pred):
        reconstruct_error = binary_crossentropy(K.flatten(true), K.flatten(pred))
        reconstruct_error *= 34*22
        return(reconstruct_error)

    opt = keras.optimizers.Adam(lr=0.00002)

    ####add three losses together, target also: kl loss, reconstructure, properties mse
    model_targets = {
    'x_pred': x_train,
    'kl_loss': np.ones((np.shape(x_train)[0], latent_dim * 2)),
    'homo_loss': y_homo,
    'lumo_loss': y_lumo
    }
    '''
    test_targets = {
    'x_pred': x_test,
    'kl_loss': np.ones((np.shape(x_test)[0], latent_dim * 2)),
    'properties_loss': y_test
    }
    '''
    total_loss = {'x_pred': reconstruct_error, 'kl_loss': kl_loss, 'homo_loss': mse_loss, 'lumo_loss': mse_loss}
    total_loss_weight = {'x_pred': 0.5*0.5, 'kl_loss': 0.5*0.5,
     'homo_loss': 0.5, 'lumo_loss': 0.5}

    #train
    total_model.compile(optimizer=opt, loss = total_loss, loss_weights = total_loss_weight)
    #autoencoder.fit(train_images, train_images)
    '''
    vae.fit(x_train, x_train,
                    epochs=120,
                    batch_size=250,
                    shuffle=True)
                    #validation_data=(x_test, x_test))
                    '''
    his = total_model.fit(x_train, model_targets,
                    epochs= 100,
                    batch_size=250,
                    #epochs= 1,
                    #batch_size=1000,
                    validation_split=0.2,
                    shuffle=True,
                    verbose = 1)
                    #validation_data=(x_test, test_targets))
    his_data = pd.DataFrame.from_dict(his.history)
    his_data.to_csv('ep100_holu_lr2_valid_hist.csv')
                    #callbacks=callbacks_list) 
    total_model.save("ep100_holu_lr2_total.h5")
    encoder.save("ep100_holu_lr2_encoder.h5")
    decoder.save("ep100_holu_lr2_decoder.h5")
    pp_homo.save("ep100_holu_lr2_pp_homo_properties.h5")
    pp_lumo.save("ep100_holu_lr2_pp_lumo_properties.h5")

    predict_raw = decoder.predict(encoder.predict(x_train)[1])
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
    #np.mean((y_homo_test - pp_homo.predict(encoder.predict(x_test)[1]))**2)
    print('test HOMO MSE = ' + str(np.mean((y_homo_test - pp_homo.predict(encoder.predict(x_test)[1]))**2)))
    print('test LUMO MSE = ' + str(np.mean((y_lumo_test - pp_lumo.predict(encoder.predict(x_test)[1]))**2)))



    print('total time:' + str(time.time() - start_time) + 'sec')
if __name__ == '__main__':
	main()


