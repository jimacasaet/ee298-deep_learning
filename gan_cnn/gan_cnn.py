''' Generative Adversarial Network w/architecture recommended by Larsen et al
NABUS, Martin Roy
MACASAET, John Rufino
'''

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import os

import numpy as np
from scipy.ndimage import filters
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

def img_resize(img,rescale_size):
	h,w,c = img.shape
	img = img[20:h-20,:,:]
	# Smooth image before resize to avoid moire patterns
	scale = img.shape[0] / float(rescale_size)
	sigma = np.sqrt(scale) / 2.0
	img = filters.gaussian_filter(img, sigma=sigma)
	img = transform.resize(img, (rescale_size, rescale_size, 3), order=3, mode='reflect')
	img = (img*255).astype(np.uint8)
	return img

class DCGAN():
	def __init__(self):
		# Input shape
		self.img_rows = 64
		self.img_cols = 64
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 128

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generates imgs
		z = Input(shape=(self.latent_dim,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):
		
		model = Sequential()
		
		model.add(Dense(8*8*256, activation="relu", input_shape=(self.latent_dim,)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))	
		model.add(Reshape((8,8,256)))
		model.add(Conv2DTranspose(filters = 256, kernel_size = 5, strides = 2, padding = "same", use_bias = False))	
		model.add(BatchNormalization())
		model.add(Activation('relu'))	
		model.add(Conv2DTranspose(filters = 128, kernel_size = 5, strides = 2, padding = "same", use_bias = False))
		model.add(BatchNormalization())		
		model.add(Activation('relu'))	
		model.add(Conv2DTranspose(filters = 32, kernel_size = 5, strides = 2, padding = "same", use_bias = False))
		model.add(BatchNormalization())
		model.add(Activation('relu'))	
		model.add(Conv2D(filters = 3, kernel_size = 5, padding = "same", use_bias = False))
		model.add(Activation('tanh'))		
		model.summary()
		return model

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(filters = 32, kernel_size = 5, padding = "same", use_bias = False, input_shape=self.img_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = "same", use_bias = False))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = "same", use_bias = False))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = "same", use_bias = False))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		return model

	def train(self, directory, epochs, batch_size=128):

		X_train = []; ctr = 0 # Images
		half_batch = int(batch_size / 2)

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------
		
			X_train = []
			idx = np.random.randint(1, 202600, half_batch)
			for i in idx :
				name = str(i)
				while (len(name) < 6) : name = '0'+name
				name = name+'.jpg'
				img = io.imread(directory+name)
				img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # [-1 to 1]
				X_train.append(img)
			imgs = np.asarray(X_train)

			# Sample noise and generate a half batch of new images
			noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			# Sample generator input
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
		print("TRAINING COMPLETE! SAVING MODEL... ")	
		self.generator.save("gan_gen.h5")

if __name__ == '__main__':
	dcgan = DCGAN()
	dcgan.train(directory="../img/", epochs=500, batch_size=64)