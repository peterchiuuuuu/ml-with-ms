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
from scipy.stats import norm
from sklearn.decomposition import PCA
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
import random
import time
from rdkit.Chem.Draw import DrawingOptions

global random_seed
random_seed = int(time.time())



########model load
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
    #dist = (latent_space_train - pos)**dim
    dist = (latent_space_train - pos)**dim
    dist = np.sum(dist, axis=1)
    dist = dist**(1/dim)
    return(dist)

def load_model_data():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global homo_model, lumo_model, encoder, decoder
    homo_model = keras.models.load_model("./keras_model/holu_retrain/ep100_holu_homo_retrain_Weipp1.h5", custom_objects={'x_pred': reconstruct_error, 'kl_loss': kl_loss, 'homo_loss': mse_loss, 'lumo_loss': mse_loss})
    lumo_model = keras.models.load_model("./keras_model/holu_retrain/ep100_holu_lumo_retrain_Weipp1.h5", custom_objects={'x_pred': reconstruct_error, 'kl_loss': kl_loss, 'homo_loss': mse_loss, 'lumo_loss': mse_loss})
    encoder = keras.models.load_model("./keras_model/holu_retrain/ep100_holu_encoder_retrain_Weipp1.h5", custom_objects={'x_pred': reconstruct_error, 'kl_loss': kl_loss, 'homo_loss': mse_loss, 'lumo_loss': mse_loss})
    decoder = keras.models.load_model("./keras_model/holu_retrain/ep100_holu_decoder_retrain_Weipp1.h5", custom_objects={'x_pred': reconstruct_error, 'kl_loss': kl_loss, 'homo_loss': mse_loss, 'lumo_loss': mse_loss})
    decoder._make_predict_function()
    #####data

    def add_space(raw_data, input_dim = 34):
        out = []    
        for i in raw_data:
            if len(i) < input_dim:
                out.append(i+' '*(input_dim - len(i)))
            else:
                out.append(i)           
        return(out)
    global out
    out_raw = pd.read_csv('./QM9_data/QM9_smiles.csv', header = None)[0] 
    out = add_space(out_raw)

    QM9_hot = []
    for i in out:
        QM9_hot.append(smiles_encoder(i))
    
    #properties
    QM9_properties = pd.read_csv('./QM9_data/QM9_properties.csv') 
    QM9_properties.columns = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
    'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    global x_train, y_homo, y_lumo
    x_train = QM9_hot
    x_train = np.reshape(x_train, (len(x_train), 34, 22))
    y_homo = QM9_properties['homo']
    y_homo = np.reshape(y_homo, (len(y_homo)))
    y_lumo = QM9_properties['lumo']
    y_lumo = np.reshape(y_lumo, (len(y_lumo)))

    global latent_space_train
    latent_space_train= encoder.predict(x_train)[0][:,:156]
    '''
    global lowest #the location of lowest y_train
    lowest = np.argsort(y_train)[0]

    global latent_space_dis
    latent_space_dis = dist_cal(latent_space_train, latent_space_train[lowest])

    global latent_space_train_seq
    latent_space_train_seq = np.argsort(latent_space_dis)
    '''
    global pca, latent_train_list
    pca = PCA(n_components = 2)
    pca_latent = pca.fit_transform(latent_space_train)
    latent_train_list = pd.DataFrame(data=pca_latent, columns = ['PC1', 'PC2'])

    global prediction_homo, prediction_lumo
    prediction_homo = homo_model.predict(encoder.predict(x_train)[1])
    prediction_lumo = lumo_model.predict(encoder.predict(x_train)[1])

    

def value_func(homo_desire, lumo_desire, y_homo, y_lumo, std = 0.1):
    ####using true properties value to cal value_func
    ##normalize to 0 - 1
    homo_value_fun = norm.pdf(y_homo, homo_desire, std)/norm.pdf(homo_desire, homo_desire, std)
    lumo_value_fun = norm.pdf(y_lumo, lumo_desire, std)/norm.pdf(lumo_desire, lumo_desire, std)
    return(homo_value_fun*lumo_value_fun)

def ran_sphere(n = 10000, pos = np.array([0, 0]), dis = 5, dim = 156):
    np.random.seed(random_seed)
    random.seed(random_seed)
    

    outt = np.array([None]*dim)
    for i in range(n):
        #d = 2
        u = np.random.normal(0,1,dim)  # an array of d normally distributed random variables
        norm=np.sum(u**2) **(0.5)
        r = random.random()**(1/dim)
        x= dis*r*u/norm
        x = x + pos
        outt = np.vstack((outt, x))
    outt = np.delete(outt, 0, 0)
    return(outt)
'''
def smiles_to_svg(smiles):
    molecule = MolFromSmiles(smiles)
    Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
 '''


########webbbbbb
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
model = None



@app.route("/")
def index():    
    #name = request.args['name']
    return render_template("request_holu.html"); 

@app.route('/result/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        homo_desire = float(request.form['homo_desire'])
        lumo_desire = float(request.form['lumo_desire'])
        ramdom_sample_value = int(request.form['ramdom_sample_value'])
        dis = int(request.form['dis'])
        std = float(request.form['std'])


        pd_desire, predict_list, opti, nei_t5, highest_desire_value_index = calculation(homo_desire, lumo_desire, ramdom_sample_value, dis, std)
        
        
        return render_template("result.html",
            homo_desire = homo_desire,
            lumo_desire = lumo_desire,
            ramdom_sample_value = ramdom_sample_value,
            dis = dis,
            std =std,
            predict_list = predict_list,
            highest_desire_value_index = highest_desire_value_index,
            out = out,
            y_homo = y_homo,
            y_lumo = y_lumo
            )

#@app.route("/calculation/<float(signed=True):homo_desire>/<float(signed=True):lumo_desire>/<int:ramdom_sample_value>/<int:dis>/<float:std>")
def calculation(homo_desire, lumo_desire, ramdom_sample_value, dis, std): 
    #pd_desire: prediction of desired space. contain ['PCA1', 'PCA2', 'desire_homo_prediction', 'desire_lumo_prediction', 'smiles', 'final_prediction']
    #predict_list: The prediction from the pd_desire in desired Eudicean distance
    desire_value = value_func(homo_desire, lumo_desire, y_homo = y_homo, y_lumo = y_lumo, std = std)
    
    highest_desire_value_index = np.argsort(desire_value)[-1]

    desire_space = ran_sphere(n = ramdom_sample_value, pos = latent_space_train[highest_desire_value_index], dis = dis).reshape(-1,156)

    decode_m = decoder.predict(desire_space)
    desire_homo_prediction = homo_model.predict(desire_space)
    desire_lumo_prediction = lumo_model.predict(desire_space)

    predict_st = []
    for i in decode_m:
        predict_st.append(smiles_decoder(i))
    predict_dict_uniq = np.unique(predict_st)

    ###
    valid_molecular = []
    for i, x in enumerate(predict_dict_uniq):
        #print(x)
        m = Chem.MolFromSmiles(x, sanitize=False)
        if m is None:
            valid_molecular.append('XXXXX')
        else:
            valid_molecular.append(x)
    ###
    final_prediction = []
    predict_dict = { i : 0 for i in valid_molecular }
    for i in predict_st:
        if i in predict_dict:
            predict_dict[i] += 1
            final_prediction.append(i)
        else:
            predict_dict['XXXXX'] += 1
            final_prediction.append('XXXXX')

    opti = pca.transform(latent_space_train[highest_desire_value_index].reshape(1,156))
    nei_t5 = pca.transform(desire_space)

    
    pd_desire = pd.DataFrame(data=nei_t5, columns = ['PCA1', 'PCA2'])
    pd_desire = pd.concat([pd_desire, pd.DataFrame(desire_homo_prediction), pd.DataFrame(desire_lumo_prediction), pd.DataFrame(predict_st), pd.DataFrame(final_prediction), pd.DataFrame(desire_value)], axis = 1)
    pd_desire.columns = ['PCA1', 'PCA2', 'desire_homo_prediction', 'desire_lumo_prediction', 'smiles', 'final_prediction', 'desire_value']
    '''
    for i in predict_dict:
        predict_dict[i] /= ramdom_sample_value
    '''
    #smiles_gp = pd_desire.groupby(['smiles'])

    for i in predict_dict:
        count_num = predict_dict[i]
        predict_dict[i] = [round(predict_dict[i]/ramdom_sample_value, 3)]
        if i in out:
            y_homo_tem = y_homo[out.index(i)]
            y_lumo_tem = y_lumo[out.index(i)]
        else:
            y_homo_tem = None
            y_lumo_tem = None
        pre_homo_tem_mean = pd_desire[pd_desire['smiles'] == i]['desire_homo_prediction'].mean()
        pre_homo_tem_std = pd_desire[pd_desire['smiles'] == i]['desire_homo_prediction'].std()
        pre_lumo_tem_mean = pd_desire[pd_desire['smiles'] == i]['desire_lumo_prediction'].mean()
        pre_lumo_tem_std = pd_desire[pd_desire['smiles'] == i]['desire_lumo_prediction'].std()
        desire_value_mean = pd_desire[pd_desire['smiles'] == i]['desire_value'].mean()
        desire_value_std = pd_desire[pd_desire['smiles'] == i]['desire_value'].std()
        predict_dict[i].extend([y_homo_tem, y_lumo_tem, pre_homo_tem_mean, pre_homo_tem_std, pre_lumo_tem_mean, pre_lumo_tem_std, count_num, desire_value_mean, desire_value_std])
    
    predict_list = []
    for key, value in predict_dict.items():
        temp = [key,value]
        predict_list.append(temp)
    predict_list = sorted(predict_list, key=lambda l:l[1][7], reverse=True) ##sort by count number
    

    #print(predict_list)
    return (pd_desire, predict_list, opti, nei_t5, highest_desire_value_index);


@app.route("/plot_smiles/<string:smile_name>")

def plot_smiles(smile_name):


    DrawingOptions.atomLabelFontSize = 5

    mnm = Chem.MolFromSmiles(smile_name, sanitize=False)
    
    fig1 = Draw.MolToMPL(mnm, fitImage=True)
    #, fitImage=True, bbox_inches='tight'
    ax = fig1.gca()
    ax.set_facecolor('w')

    coord_min = 0
    coord_max = 2.5
    ax.set_xlim((coord_min, coord_max))
    ax.set_ylim((coord_min, coord_max))
    '''
    major_ticks = np.linspace(coord_min,coord_max,6)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid()
    '''

    output1 = io.BytesIO()
    FigureCanvasAgg(fig1).print_png(output1)
    return Response(output1.getvalue(), mimetype="image/png")
    

@app.route("/plot_png/<float(signed=True):homo_desire>/<float(signed=True):lumo_desire>/<int:ramdom_sample_value>/<int:dis>/<float:std>")
def plot_png(homo_desire, lumo_desire, ramdom_sample_value, dis, std):
    
    pd_desire, predict_list, opti, nei_t5, highest_desire_value_index = calculation(homo_desire, lumo_desire, ramdom_sample_value, dis, std)
    
    groups = pd_desire.groupby('final_prediction')

    ###plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 16))
    ax1 = fig.add_subplot(111)
    sc = ax1.scatter(latent_train_list['PC1'], latent_train_list['PC2'], c = prediction_homo, cmap = 'winter', vmin = -0.43, vmax = -0.1)
    
    
    for name, group in groups:

        
        if name != 'XXXXX':
        #ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
            ax1.scatter(group['PCA1'], group['PCA2'], marker="$\u25A1$", s = 150, label=name)

    ax1.scatter(opti[0,0], opti[0,1], c = 'black', marker = 'x', s = 150) #-0.337
    
    ax1.annotate(round(float(prediction_homo[highest_desire_value_index][0]), 4), (opti[0,0], opti[0,1]), horizontalalignment='right',color='black') 
    #for i in range(int(ramdom_sample_value)):
    #    ax1.scatter(nei_t5[i,0], nei_t5[i,1], c = predict_st_pd, marker = 'p', s = 150)
        #ax1.annotate(nei_true5_val[i], (nei_t5[i,0], nei_t5[i,1]), horizontalalignment='right',color='black') 
    ax1.legend()
    #ax1.relim()
    ax1.set_title('prediction HOMO (in Hartree)')
    ax1.grid(ls = '--')
    fig.colorbar(sc)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    
    #plt.savefig('./static/test2.png')
    #plt.close(fig)
    
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

'''
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


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