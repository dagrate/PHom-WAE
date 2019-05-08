
# coding: utf-8

# # TOPWGAN, TOPWGAN with penalty and more 
# 
# Here we analyze the topological features of the generated data using WGAN, WGAN with penalty, VAE and WAE. 
# - We train and test the data for the four models ;
# - We save the generated samples for R TDA package ;
# 
# The calibation of the gradient penalty and the different parameters has been achieved in TopWGAN.

# In[26]:


#import sys
#import types
import pandas as pd
#from botocore.client import Config
#import ibm_boto3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from pylab import rcParams
#%matplotlib notebook

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix)
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector, Lambda
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.layers.merge import _Merge
from keras import backend as K
from functools import partial
from keras.metrics import binary_crossentropy

import tensorflow as tf


# # DATA HANDLER
df = pd.read_csv("creditcard.csv")
df.head()

# ### WGAN FUNCTIONS

# In[30]:


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

    
def gradient_penalty_loss(y_true, y_pred, averaged_samples, lamba_reg):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = lamba_reg * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


# In[31]:


def generate_samples(generator_model, noise_dim, num_samples=1000):
    return generator_model.predict(np.random.rand(num_samples, noise_dim))


def generate_images2D(generator_model, noise_dim, num_samples=1000):
    predicted_samples = generator_model.predict(np.random.rand(num_samples, noise_dim))

    pl.figure(figsize=(10,10))
    pl.scatter(X_train[:,0], X_train[:,1], s = 40, alpha=0.2, edgecolor = 'k', marker = '+',label='original samples') 
    pl.scatter(predicted_samples[:,0], predicted_samples[:,1],s = 10, alpha=0.9,c='r', edgecolor = 'k', marker = 'o',label='predicted') 
    pl.xticks([], [])
    pl.yticks([], [])
    pl.legend(loc='best')
    pl.tight_layout()    
    pl.show()
    
    return predicted_samples


def generate_images3D(generator_model, noise_dim, num_samples=1000):
    predicted_samples = generator_model.predict(np.random.rand(num_samples, noise_dim))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xt = X_train[:,0]
    yt = X_train[:,1]
    zt = X_train[:,2]
    pl.scatter(xt, yt, zt, c='b', alpha=0.2, marker = '+', label='original samples')
    
    xp = predicted_samples[:,0]
    yp = predicted_samples[:,1]
    zp = predicted_samples[:,2]
    pl.scatter(xp, yp, zp, c='r', alpha=0.9, marker = 'o', label='predicted')
    
    pl.xticks([], [])
    pl.yticks([], [])
    pl.legend(loc='best')
    pl.tight_layout()
    pl.show()
    
    return predicted_samples


def writetocsv(mtrx, flnm):
    """Save the samples for TDA with R (2nd notebook). We do not differentiate frauds from normal transactions"""
    dtfrm = pd.DataFrame(mtrx)
    dtfrm.to_csv(flnm, sep=',', index=None, header=None)


# ## STANDARD WGAN

# In[32]:


BATCH_SIZE = 128
TRAINING_RATIO = 5

# # AUTO-ENCODERS
# # ---------------------------

# In[38]:


LAYERS_DIM_VAE = [29, 29] #[16, 2]


# In[39]:


def evaluationplot(history, strng):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(strng + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    
def rocplot(fpr, tpr, roc_auc, strng):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('ROC ' + strng)
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()    
    

def categoricalheatmap(conf_matrix, LABELS):
        fig, ax = plt.subplots()
        im = ax.imshow(conf_matrix)
    
        ax.set_xticks(np.arange(conf_matrix.shape[0]))
        ax.set_yticks(np.arange(conf_matrix.shape[1]))
    
        trueLABELS = [str("True Class ") + s for s in LABELS]
        falseLABELS = [str("False Class ") + s for s in LABELS]
        ax.set_xticklabels(falseLABELS)
        ax.set_yticklabels(trueLABELS)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
            
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.show()
        
        
def applot(error_df):
    fig, ax = plt.subplots()
    average_precision = average_precision_score(error_df.true_class, error_df.reconstruction_error)
    precision, recall, _ = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    return average_precision


def barplot(X_train):
    n, bins, patches = plt.hist(X_train, 100, facecolor='g', alpha=0.75)
    plt.xlabel('bins')
    plt.ylabel('Probability')
    plt.title('Histogram of X')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
    
    
def scaleX(X):
    nmin = 0.0
    nmax = 1.0
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (nmax - nmin) + nmin
    return X_scaled


# In[40]:


def wasssersteinloss(ytrue, ypred):
    return K.mean(ytrue * ypred)
    

def wloss_gradpen(y_true, y_pred):
    lamba_reg = 10
    loss = K.mean(y_true * y_pred)
    gradients = K.gradients(loss, y_pred)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = lamba_reg * K.square(1 - gradient_l2_norm)
    return loss + K.mean(gradient_penalty)

    
def wloss(ytrue, ypred):
    lbda = 0.5
    loss_1 = K.mean(ytrue * ypred)
    loss_mmd = lbda * tf.sqrt( tf.square( K.mean(ytrue) - K.mean(ypred) ) )
    return loss_1 + loss_mmd
    
    
def frechet_loss(X_real, X_gen):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    
    Params:
    -- X_real   : Numpy array containing real world samples.
    -- X_gen   : Numpy array containing synthetic samples.
    
    Returns:
    --   : The Frechet Distance.
    """
    
    def tf_cov(x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx
    
    loss_norm = ( K.mean( X_real ) - K.mean( X_gen ) ) ** 2
    
    sigma_real = tf_cov( X_real )
    sigma_gen = tf_cov( X_gen )
    
    sigma_real_clip = tf.clip_by_value( sigma_real, 0.001, 1 )
    sigma_gen_clip = tf.clip_by_value( sigma_gen, 0.001, 1 )
    
    cov_xy = tf.sqrt( K.dot(sigma_real_clip, sigma_gen_clip) )
    loss_trace = tf.trace(sigma_real + sigma_gen - 2 * cov_xy )

    return loss_norm + loss_trace


# ## WAE WITH GRADIENT PENALTY
# ## ==========================

# In[78]:


def wloss(ytrue, ypred):
    lbda = 0.5
    loss_1 = K.mean(ytrue * ypred)
    loss_mmd = lbda * tf.sqrt( tf.square( K.mean(ytrue) - K.mean(ypred) ) )
    return loss_1 + loss_mmd


def wae_loss(x, x_decoded):
    
    def get_batch_size(inputs):
        return tf.cast(tf.shape(inputs)[0], tf.float32)

    def mmd_penalty(sample_qz, sample_pz):
        n = get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2
        half_size = tf.cast(half_size, dtype=tf.int32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
         
        sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
        return stat
            
    lbda = 10
    loss = K.mean(x * x_decoded)
    loss_mmd = lbda * mmd_penalty(x, x_decoded)
    #loss_mmd = lbda * tf.sqrt( tf.square( K.mean(x) - K.mean(x_decoded) ) )
    return loss + loss_mmd


# In[84]:


def waeautoencoder(X_train, layers_dim, batch_size):
    
    def sampling(args):
        z_mean = args
        #epsilon = K.random_normal(shape=(batch_size, LATENT_DIM), mean=0, stddev=1)
        return z_mean #+ K.exp(z_log_sigma) * epsilon
        
    if len(layers_dim) != 2:
        raise ValueError("Only Two Layers Allowed for VAE")
        
    INTERMEDIATE_DIM = layers_dim[0]
    LATENT_DIM = layers_dim[1] # bottleneck
    
    original_dim = X_train.shape[1]
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(INTERMEDIATE_DIM, activation='tanh')(x) # h = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    
    z_mean = Dense(LATENT_DIM)(h)
    #z_log_sigma = Dense(LATENT_DIM)(h)
    z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean])
    
    decoder_h = Dense(INTERMEDIATE_DIM, activation='tanh') # decoder_h = Dense(INTERMEDIATE_DIM, activation='relu')
    decoder_mean = Dense(original_dim, activation='tanh') # decoder_mean = Dense(original_dim, activation='sigmoid') tanh / elu / linear
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    vae = Model(x, x_decoded_mean) # end-to-end autoencoder
    encoder = Model(x, z_mean) # encoder, from inputs to latent space
    
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(LATENT_DIM,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return vae, encoder, generator, [z_mean]


# In[85]:


def scaleX(X):
    nmin = 0.0
    nmax = 1.0
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (nmax - nmin) + nmin
    return X_scaled

def calcrmse(X_train, gensamples):
    max_column = X_train.shape[1]
    rmse_lst = []
    for col in range(max_column):
        rmse_lst.append(np.sqrt(mse(X_train[:,col], gensamples[:,col])))
    return np.sum(rmse_lst) / max_column


### Data set imbalance
### ------------------
# we define different level of imbalance in the datasets for the experiments
# 0 means there is only frauds
INDX_LVL = 4
LVLS = [0, 100, 500, 1000, 5000, 10000, 25000, 40000]


NB_EPOCH = 2000
RANDOM_SEED = 42
GEN_TRANSACTIONS = 0
if GEN_TRANSACTIONS == 1:
    print("Warning. Generation of artificial transactions.")

### Preparing the data
### ------------------
#df = pd.read_csv("/mnt/hgfs/SnT/Tensors/Data/credit_card_fraud/creditcard.csv")
data = df.drop(['Time'], axis=1)
frauds = data[data.Class == 1]
normal = data[data.Class == 0]
    
cs = 2 # 2, 3, 6, 7

if cs == 7:
    data = data
elif cs == 4:
    data['Amount'] = normalize(data['Amount'].values.reshape(-1, 1))
elif cs == 5 or cs == 7:
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))


lvl = LVLS[INDX_LVL]
if lvl == 0:
    data = frauds
else:
    if GEN_TRANSACTIONS == 0:
        data = np.vstack([normal[:lvl], frauds])
        np.random.shuffle(data)
    else:
        data = np.asarray(normal)
        np.random.shuffle(data)
        data = data[:lvl]

X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
if lvl != 0:
    # Class is the last column
    y_train = X_train[:,-1]
    y_test = X_test[:,-1]
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
else:
    y_train = X_train.values[:,-1]
    X_train = X_train.values[:,:-1]
    y_test = X_test.values[:,-1]
    X_test = X_test.values[:,:-1]

if cs == 2:
    X_train = normalize(X_train)
    X_test = normalize(X_test)
elif cs == 3:
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
elif cs == 6:
    X_train = np.exp(-X_train) / np.sum(np.exp(-X_train))
    X_test = np.exp(-X_test) / np.sum(np.exp(-X_test))
elif cs == 7:
    X_train = scaleX(X_train[:,:2])
    X_test = scaleX(X_test[:,:2])

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)    
print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("max X_train[:,0]: ", np.max(X_train[:,0]))
print("max X_train[:,1]: ", np.max(X_train[:,1]))
print("min X_train[:,0]: ", np.min(X_train[:,0]))
print("min X_train[:,1]: ", np.min(X_train[:,1]))


# In[86]:


X_train_vae_len = X_train.shape[0] - (X_train.shape[0] % BATCH_SIZE)
X_test_vae_len = X_test.shape[0] - (X_test.shape[0] % BATCH_SIZE)
X_train_vae = X_train[:X_train_vae_len]
X_test_vae = X_test[:X_test_vae_len]

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("X_train_vae.shape:", X_train_vae.shape)
print("X_test_vae.shape:", X_test_vae.shape)


# In[87]:


wae, wae_encoder, wae_generator, wae_args = waeautoencoder(X_train_vae, LAYERS_DIM_VAE, BATCH_SIZE)

z_mean = wae_args[0]

wae.compile(loss=wae_loss, optimizer='RMSprop', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']) # 'accuracy', mae, mse , 'cosine_proximity'

history = wae.fit(X_train_vae, X_train_vae, epochs=NB_EPOCH, batch_size=BATCH_SIZE, shuffle=True, validation_data=(X_test_vae, X_test_vae), verbose=2)

evaluationplot(history, str(wae_loss))

predictions = wae.predict(X_test_vae, batch_size=BATCH_SIZE)
mse = np.mean(np.power(X_test_vae - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': np.mean(np.power(predictions, 2), axis=1), 'true_class': np.mean(np.power(X_test_vae, 2), axis=1)})
print( error_df.describe() )

nquant = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
store_quant = []
for i_ in nquant:
    tmp = error_df.quantile(q=i_, axis=0, numeric_only=True)
    store_quant.append(np.sqrt( (tmp[0]-tmp[1])**2 ))
print("overall rmse ", np.sum(store_quant) )

error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test[:X_test_vae_len]})


# In[88]:


# use the WAE decompression for the data generation (wae_generator)
# =================================================================

# assess visually the quality of the results
# gensamples = generate_images3D(generator, noise_dim) # 3D plot are not well rendering
# gensamples = generate_images2D(wae_generator, 29)

# results save
# writetocsv(X_train, "WAE_original.csv")
# generated = generate_samples(wae_generator, 29)
# writetocsv(generated, "WAE.csv")


# ## VAE
# ## ===

# In[47]:


def vae_loss(x, x_decoded_mean):
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

    
def vaeautoencoder(X_train, layers_dim, batch_size):
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, LATENT_DIM), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon
        
    if len(layers_dim) != 2:
        raise ValueError("Only Two Layers Allowed for VAE")
        
    INTERMEDIATE_DIM = layers_dim[0]
    LATENT_DIM = layers_dim[1] # bottleneck
    
    original_dim = X_train.shape[1]
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(INTERMEDIATE_DIM, activation='tanh')(x) # h = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    
    z_mean = Dense(LATENT_DIM)(h)
    z_log_sigma = Dense(LATENT_DIM)(h)
    z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])
    
    decoder_h = Dense(INTERMEDIATE_DIM, activation='tanh') # decoder_h = Dense(INTERMEDIATE_DIM, activation='relu')
    decoder_mean = Dense(original_dim, activation='tanh') # decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    vae = Model(x, x_decoded_mean) # end-to-end autoencoder
    encoder = Model(x, z_mean) # encoder, from inputs to latent space
    
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(LATENT_DIM,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return vae, encoder, generator, [z_mean, z_log_sigma]


# In[48]:


X_train_vae_len = X_train.shape[0] - (X_train.shape[0] % BATCH_SIZE)
X_test_vae_len = X_test.shape[0] - (X_test.shape[0] % BATCH_SIZE)
X_train_vae = X_train[:X_train_vae_len]
X_test_vae = X_test[:X_test_vae_len]

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("X_train_vae.shape:", X_train_vae.shape)
print("X_test_vae.shape:", X_test_vae.shape)


# In[59]:


vae, vae_encoder, vae_generator, vae_args = vaeautoencoder(X_train_vae, LAYERS_DIM_VAE, BATCH_SIZE)

z_mean, z_log_sigma = vae_args[0], vae_args[1]

vae.compile(optimizer='adam', loss=vae_loss, metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']) # 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'
history = vae.fit(X_train_vae, X_train_vae, epochs=NB_EPOCH, batch_size=BATCH_SIZE, shuffle=True,  validation_data=(X_test_vae, X_test_vae), verbose=2)

evaluationplot(history, str(wloss))

predictions = vae.predict(X_test_vae, batch_size=BATCH_SIZE)
mse = np.mean(np.power(X_test_vae - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': np.mean(np.power(predictions, 2), axis=1),'true_class': np.mean(np.power(X_test_vae, 2), axis=1)})
print( error_df.describe() )

nquant = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
store_quant = []
for i_ in nquant:
    tmp = error_df.quantile(q=i_, axis=0, numeric_only=True)
    store_quant.append(np.sqrt( (tmp[0]-tmp[1])**2 ))
print("overall rmse ", np.sum(store_quant) )

error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test[:X_test_vae_len]})


# In[61]:


# use the VAE decompression for the data generation (vae_generator)
# =================================================================


# assess visually the quality of the results
# gensamples = generate_images3D(generator, noise_dim) # 3D plot are not well rendering
# gensamples = generate_images2D(vae_generator, 29)

# results save
# writetocsv(X_train_vae, "VAE.csv")
# generated = generate_samples(vae_generator, noise_dim)
# writetocsv(generated, "VAE.csv")

