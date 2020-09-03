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


import pandas as pd
from IPython import display
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, UpSampling2D, UpSampling1D, MaxPooling1D
from keras.layers.recurrent import GRU
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

def main():
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

	#####nn
	input_dim = 34
	input_ = Input(shape = (input_dim,22))
	x = Conv1D(10, 5, activation='tanh', padding='same')(input_)
	x = Conv1D(5, 5, activation='tanh', padding='same')(x)
	x = Conv1D(10, 5, activation='tanh', padding='same')(x)
	x = Conv1D(22, 5, activation='softmax', padding='same')(x)
	autoencoder = Model(input_, x)
	####
	#data 
	#x_train = QM9_hot[0:10000]
	x_train = QM9_hot
	x_train = np.reshape(x_train, (len(x_train), 34, 22))
	x_test = QM9_hot[10000:10500]
	x_test = np.reshape(x_test, (len(x_test), 34, 22))
	#train

	autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')
	#autoencoder.fit(train_images, train_images)
	autoencoder.fit(x_train, x_train,
	                epochs=20,
	                batch_size=250,
	                shuffle=True)
	                #validation_data=(x_test, x_test))
	###predict
	predict_raw = autoencoder.predict(x_train)
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