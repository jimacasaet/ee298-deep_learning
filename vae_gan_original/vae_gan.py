''' VAE-GAN (Attempt at replicating Larsen et al's work)
NABUS, Martin Roy
MACASAET, John Rufino
'''

from __future__ import print_function

import numpy as np
from scipy.ndimage import filters
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Flatten, Reshape, Dropout, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import plot_model

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
optimizer = RMSprop(lr=0.0001) # VAE error reaches NaN when lr = 0.0003 in our runs
preload = False
chkpt = 2000 
''' PRELOAD OPTIONS:
OPTION A (preload=True): LOAD ALL DATA BEFORE RUNNING
OPTION B (preload=False): LOAD DATA WHILE RUNNING (slower overall but does not use up RAM for all images) 
'''

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# build encoder model
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
# plot_model(encoder, to_file='vae_gan_encoder.png', show_shapes=True)

# build decoder model
filter = 256
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
# plot_model(decoder, to_file='vae_gan_decoder.png', show_shapes=True)

# build discriminator model
filter = 32; #kernel_size = 3
disc_input = Input(shape=input_shape, name='discriminator_input')
x3 = disc_input
x3 = Conv2D(filters=filter, kernel_size=kernel_size, activation='relu', padding='same', use_bias=False)(x3)
filter *= 4
for i in range(3) :
	x3 = Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding='same', use_bias=False)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation('relu')(x3)
	if(i==0) : filter *= 2 # 128 -> 256
x3 = Flatten()(x3)
x3 = Dense(512)(x3)
x3 = BatchNormalization()(x3)
lth_layer = Activation('relu')(x3)
disc_out = Dense(1, activation='sigmoid')(lth_layer)

# instantiate discriminator model
discriminator = Model(disc_input, disc_out, name='discriminator')
discriminator.summary()

# INSTANTIATE TRAINING MODELS
# discriminator training model
discriminator_model = Model(disc_input, disc_out, name='discriminator_model')
discriminator_model.compile(loss='binary_crossentropy',optimizer=optimizer)
print("DISCRIMINATOR TRAINER:"); discriminator_model.summary()
# decoder training model (GAN)
discriminator.trainable = False
outputs = discriminator(decoder(latent_inputs))
generator_model = Model(latent_inputs, outputs, name='generator_model')
generator_model.compile(loss='binary_crossentropy', optimizer=optimizer)
print("DECODER TRAINER (GENERATOR):"); generator_model.summary()
# encoder & decoder training model (VAE)
discriminator_lth = Model(disc_input, lth_layer, name='discriminator_lth_layer')
discriminator_lth.trainable = False
outputs = discriminator_lth(decoder(encoder(inputs)[2]))
vae_model = Model(inputs, outputs, name='vae_model')
def encoder_loss(y_true,y_pred):
	lth_loss = 0.5 * metrics.mse(K.flatten(y_true), K.flatten(y_pred))
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return K.mean(lth_loss + kl_loss)
vae_model.compile(loss=encoder_loss, optimizer=optimizer)
print("ENCODER&DECODER TRAINER (VAE):"); vae_model.summary()

if(len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == 'continue')) :
	if(len(sys.argv) > 1 and sys.argv[1] == 'continue'):
		print("Loading weights...")
		encoder.load_weights('vae_gan_enc.h5')
		decoder.load_weights('vae_gan_dec.h5')
		discriminator.load_weights('vae_gan_disc.h5')
		
	X_train = []; directory = "../img/"; ctr = 0
	
	if(preload) : # X_train will hold ALL images
		X_train = []; ctr = 0
		for name in sorted(os.listdir(directory)):
			img = io.imread(directory+name)
			img = img_resize(img,64)
			img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # tanh
			X_all.append(img)
			ctr = ctr+1;
			if(ctr%10000 == 0) : print(ctr,"images loaded!")
		print("All images loaded!")
		X_train = np.asarray(X_train)
	
	half_batch = int(batch_size/2)
	
	# TRAINING PHASE
	for epoch in range(epochs) :
		idx = np.random.randint(1, 202600, batch_size) # randomized minibatch
		if(preload) :
			idx = idx-1; imgs = X_train[idx]
		else : # no preload
			X_train = []
			for i in idx :
				name = str(i)
				while (len(name) < 6) : name = '0'+name
				name = name+'.jpg'
				img = io.imread(directory+name)
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # tanh
				X_train.append(img)
			imgs = np.asarray(X_train)
			idx = idx-1; 
		
		# DISCRIMINATOR TRAINING
		disc_loss1 = discriminator_model.train_on_batch(imgs, np.ones((batch_size,1)))
		lcode = encoder.predict(imgs)[2]
		lc_img = decoder.predict(lcode)
		disc_loss2 = discriminator_model.train_on_batch(lc_img, np.zeros((batch_size,1)))
		noise = np.random.normal(0, 1, (batch_size, latent_dim))
		gen_imgs = decoder.predict(noise)
		disc_loss3 = discriminator_model.train_on_batch(gen_imgs, np.zeros((batch_size,1)))
		
		# GENERATOR TRAINING
		idx = np.random.randint(1, 202600, half_batch) # randomized minibatch
		if(preload) : 
			idx = idx-1;
			imgs = X_train[idx]
		else :
			X_train = []
			for i in idx :
				name = str(i)
				while (len(name) < 6) : name = '0'+name
				name = name+'.jpg'
				img = io.imread(directory+name)
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # tanh
				X_train.append(img)
			imgs = np.asarray(X_train)
			
		latent_pred = encoder.predict(imgs)[2]
		gen_loss1 = generator_model.train_on_batch(latent_pred, np.ones((half_batch,1)))
		noise = np.random.normal(0, 1, (half_batch, latent_dim))
		gen_loss2 = generator_model.train_on_batch(noise, np.ones((half_batch,1)))
		
		# VAE TRAINING
		idx = np.random.randint(1, 202600, batch_size) # randomized minibatch
		if(preload) : 
			idx = idx-1;
			imgs = X_train[idx]
		else :
			X_train = []
			for i in idx :
				name = str(i)
				while (len(name) < 6) : name = '0'+name
				name = name+'.jpg'
				img = io.imread(directory+name)
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # tanh
				X_train.append(img)
			imgs = np.asarray(X_train)
		imgs_lth = discriminator_lth.predict(imgs)
		enc_loss = vae_model.train_on_batch(imgs, imgs_lth)
		# checkpointing
		if((epoch+1)%chkpt==0 and epoch < epochs) :
			print("Checkpoint at",epoch+1,": saving model weights...")
			encoder.save_weights('vae_gan_enc.h5')
			decoder.save_weights('vae_gan_dec.h5')
			discriminator.save_weights('vae_gan_disc.h5')
		# loss display
		print("EPOCH",epoch+1)
		print("Discriminator losses:",disc_loss1,disc_loss2,disc_loss3)
		print("Generator losses:",gen_loss1,gen_loss2)
		print("VAE losses:",enc_loss)

	print("Training finished! Saving weights...")
	encoder.save_weights('vae_gan_enc.h5')
	decoder.save_weights('vae_gan_dec.h5')
	discriminator.save_weights('vae_gan_disc.h5')
	
elif(sys.argv[1] == "test") :
	encoder.load_weights('vae_gan_enc.h5')
	decoder.load_weights('vae_gan_dec.h5')
	name = input("Input name of file (e.g. 000001.jpg) or q to quit... ")
	while(name != 'q') :
		imgt = io.imread("../img/"+name)
		imgt = img_resize(imgt,64)
		img_test = []
		img_test.append((np.asarray(imgt) - 127.5) / 127.5) # tanh
		img_test = np.asarray(img_test)
		lcode = encoder.predict(img_test)[2]
		res = decoder.predict(lcode)
		res = np.uint8(res*127.5 + 127.5) # tanh
		plt.subplot(121), plt.imshow(imgt)
		plt.subplot(122), plt.imshow(res[0])
		plt.show()
		name = input("Input name of file (e.g. 000001.jpg) or q to quit... ")
		
elif(sys.argv[1] == "generate") :
	decoder.load_weights('vae_gan_dec.h5')
	name = "hi"
	while(name != 'q') :
		noise = noise = np.random.normal(0, 1, (1, latent_dim))
		res = decoder.predict(noise)
		res = np.uint8(res*127.5 + 127.5) # tanh
		plt.imshow(res[0]), plt.show()
		name = input("Type in anything to generate another image or q to quit... ")
