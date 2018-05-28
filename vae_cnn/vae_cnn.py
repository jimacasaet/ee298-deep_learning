''' Variational Autoencoder w/architecture recommended by Larsen et al
NABUS, Martin Roy
MACASAET, John Rufino
'''
from __future__ import print_function

import numpy as np
from scipy.ndimage import filters
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Flatten, Reshape, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils import plot_model
from keras.optimizers import RMSprop

import os
import sys

def img_resize(img,rescale_size):
	h,w,c = img.shape
	img = img[20:h-20,:]
	# Smooth image before resize to avoid moire patterns
	scale = img.shape[0] / float(rescale_size)
	sigma = np.sqrt(scale) / 2.0
	img = filters.gaussian_filter(img, sigma=sigma)
	img = transform.resize(img, (rescale_size, rescale_size, 3), order=3, mode='reflect')
	img = (img*255).astype(np.uint8)
	return img

batch_size = 64
input_shape = (64,64,3)
latent_dim = 128
epochs = 10000	# Treats one epoch as a per-minibatch run instead of one pass on all data
epsilon_std = 1.0
kernel_size = 5
filter = 32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(3) :
	filter *= 2 # 64, 128, 256
	x = Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
sh = K.int_shape(x)
x = Flatten()(x)
x = Dense(2048)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)	
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x2 = Dense(sh[1] * sh[2] * sh[3])(latent_inputs)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)	
x2 = Reshape((sh[1], sh[2], sh[3]))(x2)
for i in range(3) :
	x2 = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=2, padding='same', use_bias=False)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	filter //= 2 # Filter before division: 256, 128, 32
	if(i == 1) : filter //= 2 # 64 -> 32	
outputs = Conv2D(filters=3, kernel_size=kernel_size, activation='tanh', padding='same', name='decoder_output', use_bias=False)(x2)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)
# compute VAE loss
xent_loss = 64*64 * metrics.mse(K.flatten(inputs), K.flatten(outputs))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
optimizer = RMSprop(lr=0.0003)
vae.compile(optimizer=optimizer)
vae.summary()
# plot_model(vae, to_file='vae.png', show_shapes=True)

if(len(sys.argv) == 1) :
	X_train = []; directory = "./img/"; ctr = 0

	for epoch in range(epochs) :
		X_train = []
		idx = np.random.randint(1, 202600, batch_size) # randomized minibatch
		for i in idx :
			name = str(i)
			while (len(name) < 6) : name = '0'+name
			name = name+'.jpg'
			img = io.imread(directory+name)
			img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 
			X_train.append(img)
		imgs = np.asarray(X_train)
		run_loss = vae.train_on_batch(imgs, None)
		print(epoch,"Loss =",run_loss)
	print("Training finished!")
	
	vae.save_weights('vae_cnn.h5')
	# decoder.save_weights('vae_decoder.h5')
	
elif(sys.argv[1] == "test") :
	vae.load_weights('vae_cnn.h5')
	name = input("Input name of file (e.g. 000001.jpg) or q to quit... ")
	while(name != 'q') :
		imgt = io.imread("./img/"+name)
		imgt = img_resize(imgt,64)
		img_test = []
		img_test.append((np.asarray(imgt) - 127.5) / 127.5) 
		img_test = np.asarray(img_test)
		res = vae.predict(img_test)
		res = np.uint8(res*127.5 + 127.5)
		plt.subplot(121), plt.imshow(imgt)
		plt.subplot(122), plt.imshow(res[0])
		plt.show()
		name = input("Input name of file (e.g. 000001.jpg) or q to quit... ")
		
elif(sys.argv[1] == "generate") :
	vae.load_weights('vae_cnn.h5')
	name = "hi"
	while(name != 'q') :
		noise = noise = np.random.normal(0, 1, (1, latent_dim))
		res = decoder.predict(noise)
		res = np.uint8(res*127.5 + 127.5) 
		plt.imshow(res[0]), plt.show()
		name = input("Type in q to quit... ")
