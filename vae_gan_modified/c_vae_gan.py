''' Conditional VAE-GAN with Modified Discriminator
NABUS, Martin Roy
MACASAET, John Rufino
'''

from __future__ import print_function

import numpy as np
from scipy.ndimage import filters
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, concatenate, Lambda, BatchNormalization, Activation, Flatten, Reshape, Dropout, UpSampling2D
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
label_shape = (40,)
latent_dim = 128	
epochs = 5000	# Treats one epoch as a per-minibatch run instead of one pass on all data
epsilon_std = 1.0
kernel_size = 5
filter = 32
gamma = 0.75
optimizer = RMSprop(lr=0.0001)	# VAE error reaches NaN when lr = 0.0003 in our runs
preload = False
chkpt = 5000
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
input_labels = Input(shape=label_shape, name='input_labels')
input_lbl = Dense(64*64, activation='tanh')(input_labels)
input_lbl = Reshape((64,64,1))(input_lbl)
x = concatenate([inputs, input_lbl])
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
encoder = Model([inputs, input_labels], [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='cvg_encoder.png', show_shapes=True)

# build decoder model
filter = 256
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x2 = Dense(sh[1] * sh[2] * sh[3])(latent_inputs)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)	
x2 = Reshape((sh[1], sh[2], sh[3]))(x2)
for i in range(3) :
	x2 = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=2, padding='same')(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	filter //= 2 # Filter before division: 256, 128, 32
	if(i == 1) : filter //= 2 # 64 -> 32	
outputs = Conv2D(filters=3, kernel_size=kernel_size, activation='tanh', padding='same', name='decoder_output', use_bias=False)(x2)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
# plot_model(decoder, to_file='cvg_decoder.png', show_shapes=True)

# build discriminator model
filter = 32; kernel_size = 3
disc_input = Input(shape=input_shape, name='discriminator_input')
disc_labels = Input(shape=label_shape, name='discriminator_labels')
disc_lbl = Dense(64*64, activation='tanh')(disc_labels)
disc_lbl = Reshape((64,64,1))(disc_lbl)
x3 = concatenate([disc_input, disc_lbl])
x3 = Conv2D(filters=filter, kernel_size=kernel_size, padding='same', use_bias=False)(x3)
x3 = LeakyReLU(alpha=0.2)(x3)
x3 = Dropout(0.25)(x3)
filter *= 2
for i in range(2) :
	x3 = Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding='same', use_bias=False)(x3)
	x3 = LeakyReLU(alpha=0.2)(x3)
	x3 = Dropout(0.25)(x3)
	x3 = BatchNormalization()(x3)
	filter *= 2 # 64,128,256
x3 = Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding='same')(x3)
x3 = LeakyReLU(alpha=0.2)(x3)
x3 = Dropout(0.25)(x3)
x3 = Flatten()(x3)
disc_out = Dense(1, activation='sigmoid')(x3)

# instantiate discriminator model
discriminator = Model([disc_input,disc_labels], disc_out, name='discriminator')
# discriminator.summary()

# INSTANTIATE TRAINING MODELS
# discriminator training model
discriminator_model = Model([disc_input,disc_labels], disc_out, name='discriminator_model')
discriminator_model.compile(loss='binary_crossentropy',optimizer=optimizer)
print("DISCRIMINATOR TRAINER:"); discriminator_model.summary()
# decoder training model (GAN)
discriminator.trainable = False
outputs = discriminator([decoder(latent_inputs),disc_labels])
generator_model = Model([latent_inputs,disc_labels], outputs, name='generator_model')
generator_model.compile(loss='binary_crossentropy', optimizer=optimizer)
print("DECODER TRAINER (GENERATOR):"); generator_model.summary()
# encoder & decoder training model (VAE)
outputs = decoder(encoder([inputs,input_labels])[2])
vae_model = Model([inputs,input_labels], outputs)
xent_loss = 64*64*metrics.mse(K.flatten(inputs), K.flatten(outputs))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae_model.add_loss(vae_loss)
vae_model.compile(optimizer=optimizer)
print("ENCODER&DECODER TRAINER (VAE):"); vae_model.summary()

if(len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == 'continue')) :
	if(len(sys.argv) > 1 and sys.argv[1] == 'continue') :
		encoder.load_weights('c_vae_gan_enc.h5')
		decoder.load_weights('c_vae_gan_dec.h5')
		discriminator.load_weights('c_vae_gan_disc.h5')
		
	X_train = []; directory = "../img/"; ctr = 0
	
	# LOAD ALL ATTRIBUTE VECTORS OF TRAINING SET
	X_labels = []
	f = open("../list_attr_celeba.txt",'r')
	for line in f :
		if(ctr < 2) : # Skip first two lines
			ctr = ctr+1; continue;
		l = [float(x) for x in line.split()[1:]]
		X_labels.append(l)
	X_labels = np.asarray(X_labels)
	
	if(preload) : # X_train will hold ALL images
		X_train = []; ctr = 0
		for name in sorted(os.listdir(directory)):
			img = io.imread(directory+name)
			img = img_resize(img,64)
			img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 
			X_all.append(img)
			ctr = ctr+1;
			if(ctr%10000 == 0) : print(ctr,"images loaded!")
		print("All images loaded!")
		X_train = np.asarray(X_train)
	
	half_batch = int(batch_size/2)
	
	# TRAINING PHASE
	for epoch in range(epochs) :
		idx = np.random.randint(1, 202600, half_batch) # randomized minibatch
		if(preload) :
			idx = idx-1; imgs = X_train[idx]
		else : # image loading for no preload
			X_train = []
			for i in idx :
				name = str(i)
				while (len(name) < 6) : name = '0'+name
				name = name+'.jpg'
				img = io.imread(directory+name)
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 
				X_train.append(img)
			imgs = np.asarray(X_train)
			idx = idx-1; 
		lbls = X_labels[idx]
		
		# DISCRIMINATOR TRAINING
		disc_loss1 = discriminator_model.train_on_batch([imgs,lbls], np.ones((half_batch,1)))
		lcode = encoder.predict([imgs,lbls])[2]
		lc_img = decoder.predict(lcode)
		disc_loss2 = discriminator_model.train_on_batch([lc_img,lbls], np.zeros((half_batch,1)))
		noise = np.random.normal(0, 1, (half_batch, latent_dim))
		gen_imgs = decoder.predict(noise)
		disc_loss3 = discriminator_model.train_on_batch([gen_imgs,lbls], np.zeros((half_batch,1)))
		
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
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 
				X_train.append(img)
			imgs = np.asarray(X_train)
			idx = idx-1;  
		lbls = X_labels[idx]
		latent_pred = encoder.predict([imgs,lbls])[2]
		gen_loss1 = generator_model.train_on_batch([latent_pred,lbls], np.ones((half_batch,1)))
		noise = np.random.normal(0, 1, (half_batch, latent_dim))
		gen_loss2 = generator_model.train_on_batch([noise,lbls], np.ones((half_batch,1)))
		
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
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 
				X_train.append(img)
			imgs = np.asarray(X_train)
			idx = idx-1; 
		lbls = X_labels[idx]
		enc_loss = vae_model.train_on_batch([imgs,lbls], None)
		# checkpointing 
		print("EPOCH",epoch+1)
		print("Discriminator losses:",disc_loss1,disc_loss2,disc_loss3)
		print("Generator losses:",gen_loss1,gen_loss2)
		print("VAE losses:",enc_loss)
		# loss display
		if((epoch+1)%chkpt==0 and epoch < epochs) :
			print("Checkpoint at",epoch,": saving model weights...")
			encoder.save_weights('c_vae_gan_enc.h5')
			decoder.save_weights('c_vae_gan_dec.h5')
			discriminator.save_weights('c_vae_gan_disc.h5')
			
	print("Training finished! Saving weights...")
	encoder.save_weights('c_vae_gan_enc.h5')
	decoder.save_weights('c_vae_gan_dec.h5')
	discriminator.save_weights('c_vae_gan_disc.h5')
	
elif(sys.argv[1] == "test") :
	encoder.load_weights('c_vae_gan_enc.h5')
	decoder.load_weights('c_vae_gan_dec.h5')
	name = input("Input name of file (e.g. 000001.jpg) or q to quit... ") 
	while(name != 'q') :
		vect = input("Enter attribute vector: ")
		imgt = io.imread("../img/"+name)
		avec = [float(x) for x in vect.split()]
		avecs = []; avecs.append(avec); avecs = np.asarray(avecs)
		imgt = img_resize(imgt,64)
		img_test = []; img_test.append((np.asarray(imgt) - 127.5) / 127.5) 
		img_test = np.asarray(img_test)
		lcode = encoder.predict([img_test,avecs])[2]
		res = decoder.predict(lcode)
		res = np.uint8(res*127.5 + 127.5) 
		plt.subplot(121), plt.imshow(imgt)
		plt.subplot(122), plt.imshow(res[0])
		plt.show()
		name = input("Input name of file (e.g. 000001.jpg) or q to quit... ")
		
elif(sys.argv[1] == "generate") :
	decoder.load_weights('c_vae_gan_dec.h5')
	name = "hi"
	while(name != 'q') :
		noise = noise = np.random.normal(0, 1, (1, latent_dim))
		res = decoder.predict(noise)
		res = np.uint8(res*127.5 + 127.5) 
		plt.imshow(res[0]), plt.show()
		name = input("Type in anything to generate another image or q to quit... ")
