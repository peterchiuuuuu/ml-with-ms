import tensorflow as tf
from keras.models import Model
import keras

import os
import time
import random
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
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, UpSampling2D, UpSampling1D, MaxPooling1D, Lambda, Reshape
#from keras.layers import Conv1DTranspose
##from keras.layers import Conv1DTranspose
from rdkit import RDLogger  
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Flatten, RepeatVector, Dropout
from keras.losses import mse, binary_crossentropy
from keras.layers.merge import Concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
#from tensorflow.nn import conv1d_transpose
from keras import Sequential
from keras.layers.normalization import BatchNormalization
from tensorflow.python.ops.nn_ops import conv1d_transpose
from tensorflow.keras.models import model_from_json
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import tensorflow

data_route = '/Users/peterchiu/Documents/data_analysis/QM9/'
#dsgdb9nsd_000001.xyz

input_dim = 34
def add_space(raw_data, input_dim = 34):
    out = []
    for i in raw_data:
        if len(i) < input_dim:
            out.append(i+' '*(input_dim - len(i)))
        else:
            out.append(i)           
    return(out)

'''
SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u']
'''
SMILES_CHARS = [' ', 'C',
 'N',
 'O',
 '#',
 '=',
 '1',
 '(',
 ')',
 'c',
 '[',
 'n',
 'H',
 ']',
 'o',
 '3',
 '+',
 '-',
 '2',
 'F',
 '4',
 '5']
smi2index = dict( (c,i) for i,c in enumerate(SMILES_CHARS))
index2smi = dict( (i,c) for i,c in enumerate(SMILES_CHARS))
def smiles_encoder(smiles, maxlen=34):
    #print(smiles)
    #smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    #print(smiles)
    X = np.zeros((maxlen, len(SMILES_CHARS)))
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


out_raw = pd.read_csv('QM9_smiles.csv', header = None)[0]
out = add_space(out_raw)


QM9_properties = pd.read_csv('QM9_properties.csv') 
QM9_properties.columns = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']


QM9_hot = []
for i in out:
    #print(i)
    QM9_hot.append(smiles_encoder(i))

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


properties = keras.models.load_model("ep100_gap_valid_properties.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 
                                                           'kl_loss': kl_loss,
                                        
                                                                                          'mse_loss': mse_loss})
def pp_prediction(x, model):
    
    return(model.predict(x).flatten())
encoder = keras.models.load_model("ep100_gap_valid_encoder.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 
                                                           'kl_loss': kl_loss,
                                        
                                                                                          'mse_loss': mse_loss})
decoder = keras.models.load_model("ep100_gap_valid_decoder.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 
                                                           'kl_loss': kl_loss,
                                        
                                                                                          'mse_loss': mse_loss})

def ran_sphere(n = 10000, pos = np.array([0, 0]), dis = 1, dim = 156):
    import random
    out = np.array([None]*dim)
    for i in range(n):
        #d = 2
        u = np.random.normal(0,1,dim)  # an array of d normally distributed random variables
        norm=np.sum(u**2) **(0.5)
        r = random.random()**(1/dim)
        x= dis*r*u/norm
        x = x + pos
        out = np.vstack((out, x))
    out = np.delete(out, 0, 0)
    return(out)

x_train = QM9_hot
x_train = np.reshape(x_train, (len(x_train), 34, 22))
y_train = QM9_properties['gap']
y_train = np.reshape(y_train, (len(y_train)))
#x_train.shape

lowest = np.argsort(y_train)[0]


####
latent_space = encoder.predict(x_train)[1]

x_test = ran_sphere(pos = latent_space[lowest], dis = 5)

decode_m = decoder.predict(x_test)
#decode_m = decoder.predict(encoder.predict(x_train)[1])

predict_st = []
for i in decode_m:
    predict_st.append(smiles_decoder(i))

invalid = 0
oo = []
for i, x in enumerate(predict_st):
    #print(x)
    m = Chem.MolFromSmiles(x, sanitize=False)
    if m is None:
        invalid += 1
        oo.append('XXXXX')
    else:
        oo.append(x)
        '''
        try:
            Chem.SanitizeMol(m)
            oo.append(m)
        except:
            oo.append('XXXXX11')
            invalid += 1
        '''
df = pd.DataFrame(oo)

df.to_csv('decoder_m.csv', index=False)









