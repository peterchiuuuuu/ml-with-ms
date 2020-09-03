import io
import os
import sys
import random
import pandas as pd
import numpy as np
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.ioff()
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, UpSampling2D, UpSampling1D, MaxPooling1D, Lambda, Reshape
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
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sklearn.decomposition import PCA

########model load
def kl_mse_loss(true, pred):
    # Reconstruction loss
    mse_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))
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
def reconstruct_error(true, pred):
    reconstruct_error = binary_crossentropy(K.flatten(true), K.flatten(pred))
    reconstruct_error *= 34*22
    return(reconstruct_error)

SMILES_CHARS = [' ', 'C', 'N', 'O', '#', '=', '1', '(', ')', 'c', '[', 
'n', 'H', ']', 'o', '3', '+', '-', '2', 'F', '4', '5']
global smi2index, index2smi
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

def dist_cal(latent_space_train, pos, dim = 0.5):
    dist = (latent_space_train - pos)**dim
    dist = np.sum(dist, axis=1)
    dist = dist**(1/dim)
    return(dist)

def load_model_data():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global properties, encoder, decoder
    properties = keras.models.load_model("./keras_model/ep100_ppsum_valid_properties.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 'kl_loss': kl_loss,'mse_loss': mse_loss})
    encoder = keras.models.load_model("./keras_model/ep100_ppsum_valid_encoder.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 'kl_loss': kl_loss,'mse_loss': mse_loss})
    decoder = keras.models.load_model("./keras_model/ep100_ppsum_valid_decoder.h5", custom_objects={'kl_mse_loss': kl_mse_loss, 'kl_loss': kl_loss,'mse_loss': mse_loss})
    #####data

    def add_space(raw_data, input_dim = 34):
        out = []
        for i in raw_data:
            if len(i) < input_dim:
                out.append(i+' '*(input_dim - len(i)))
            else:
                out.append(i)           
        return(out)
    out_raw = pd.read_csv('./QM9_data/QM9_smiles.csv', header = None)[0] 
    out = add_space(out_raw)

    QM9_hot = []
    for i in out:
        QM9_hot.append(smiles_encoder(i))
    
    #properties
    QM9_properties = pd.read_csv('./QM9_data/QM9_properties.csv') 
    QM9_properties.columns = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
    'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    global x_train, y_train
    x_train = QM9_hot
    x_train = np.reshape(x_train, (len(x_train), 34, 22))
    y_train = QM9_properties['lumo']
    y_train = np.reshape(y_train, (len(y_train)))

    global latent_space_train
    latent_space_train= encoder.predict(x_train)[0][:,:156]
    global lowest #the location of lowest y_train
    lowest = np.argsort(y_train)[0]

    global latent_space_dis
    latent_space_dis = dist_cal(latent_space_train, latent_space_train[lowest])

    global pca, latent_train_list
    pca = PCA(n_components = 2)
    pca_latent = pca.fit_transform(latent_space_train)
    latent_train_list = pd.DataFrame(data=pca_latent, columns = ['PC1', 'PC2'])

    global p_prediction
    p_prediction = properties.predict(encoder.predict(x_train)[1])


########webbbbbb
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
model = None



@app.route("/")
def index():    
    #name = request.args['name']
    return render_template("request_distance.html"); 

@app.route('/result/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        top_num = int(request.form['top_num'])
        print(top_num)
        return render_template("result.html",top_num = top_num)
'''
@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form

    if request.method == 'POST':

        result = request.form
        top_num = result['top_num']
        out_point = []
        nei_true5_val = []
        for i in range(int(top_num)):
            out_point.append(latent_space_train[np.argsort(latent_space_dis)[i+1]])
            nei_true5_val.append(y_train[np.argsort(latent_space_dis)[i+1]])
        out_point = np.array(out_point).reshape(int(top_num), 156)
        ###min
        min_p = pca.transform(latent_space_train[lowest].reshape(1,156))
        nei_t5 = pca.transform(out_point)
        ###plot
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(8, 8))
        sc = plt.scatter(latent_train_list['PC1'], latent_train_list['PC2'], c = p_prediction, cmap = 'rainbow', vmin = -0.2, vmax = 0.2)

        plt.scatter(min_p[0,0], min_p[0,1], c = 'black', marker = 'x', s = 150) #-0.337
        plt.annotate(y_train[lowest], (min_p[0,0], min_p[0,1]), horizontalalignment='right',color='black') 
        for i in range(int(top_num)):
            plt.scatter(nei_t5[i,0], nei_t5[i,1], c = 'black', marker = 'p', s = 150    )
            plt.annotate(nei_true5_val[i], (nei_t5[i,0], nei_t5[i,1]), horizontalalignment='right',color='black') 

        plt.grid(ls = '--')
        plt.colorbar(sc)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.savefig('./static/test2.png')
        plt.close(fig)


       # return render_template("result.html")
    return render_template("result.html",result = result) 

@app.route("/matplot-as-image.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    num_x_points=50
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")
'''
@app.route("/plot_png/")
def plot_png(top_num):
    """ renders the plot on the fly.
    """
    #if request.method == 'POST':
    #top_num = result['top_num']

    #result = request.form
    #top_num = result['top_num']
    '''
    num_x_points=50
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])
    '''
    out_point = []
    nei_true5_val = []
    for i in range(int(top_num)):
        out_point.append(latent_space_train[np.argsort(latent_space_dis)[i+1]])
        nei_true5_val.append(y_train[np.argsort(latent_space_dis)[i+1]])
    out_point = np.array(out_point).reshape(int(top_num), 156)
    ###min
    min_p = pca.transform(latent_space_train[lowest].reshape(1,156))
    nei_t5 = pca.transform(out_point)
    ###plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(latent_train_list['PC1'], latent_train_list['PC2'], c = p_prediction, cmap = 'rainbow', vmin = -0.2, vmax = 0.2)

    ax.scatter(min_p[0,0], min_p[0,1], c = 'black', marker = 'x', s = 150) #-0.337
    
    ax.annotate(y_train[lowest], (min_p[0,0], min_p[0,1]), horizontalalignment='right',color='black') 
    for i in range(int(top_num)):
        ax.scatter(nei_t5[i,0], nei_t5[i,1], c = 'black', marker = 'p', s = 150    )
        ax.annotate(nei_true5_val[i], (nei_t5[i,0], nei_t5[i,1]), horizontalalignment='right',color='black') 

    ax.grid(ls = '--')
    fig.colorbar(sc)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    
    #plt.savefig('./static/test2.png')
    #plt.close(fig)
    
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response
'''

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig


@app.route("/aaa")
def aa():
    return "Hello, lala!"	
'''
    
if __name__ == "__main__": 
    load_model_data()
    app.run()