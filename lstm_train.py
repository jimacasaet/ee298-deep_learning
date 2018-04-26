from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import sys
import nltk

from gensim.models import KeyedVectors

def str_reverse(str) :
	s = str.split()
	s.reverse()
	return " ".join(s)

def input2target(data_path, sos, eos):
	input_texts = []
	target_texts = []

	with open(data_path, 'r', encoding='utf-8') as f:
		lines = f.read().split('\n')
	for line in lines:
		if len(line) <= 0: continue
		line = line.replace(",", " , ")
		line = line.replace(".", " . ")
		line = line.replace("!", " ! ")
		line = line.replace("?", " ? ")
		line = line.replace("\"", " \" ")
		line = line.replace("(", " ( ")
		line = line.replace(")", " ) ")
		# line = line.replace(":", " : ")
		line = line.replace("/", " / ")
		line = line.replace("[", " [ ")
		line = line.replace("]", " ] ")
		line = line.lower()
		target_text, input_text = line.split('\t') # TGL-EN
		# input_text, target_text = line.split('\t') # EN-TGL
		# input_text = str_reverse(input_text) # reverse input
		# print(input_text , " : ", target_text)
		target_text = "%s %s %s" % (sos, target_text, eos)
		input_texts.append(input_text)
		target_texts.append(target_text)

	return input_texts, target_texts

def get_all_files(data_dir, sos, eos) :
	input_texts_all = []; input_texts = []
	target_texts_all = []; target_texts = []
	for name in sorted(os.listdir(data_dir)):
		path = os.path.join(data_dir, name)
		input_texts, target_texts = input2target(path, sos, eos)
		if(input_texts_all == []): input_texts_all = input_texts
		else : input_texts_all = input_texts_all + input_texts
		if(target_texts_all == []): target_texts_all = target_texts
		else : target_texts_all = target_texts_all + target_texts
	
	return input_texts_all, target_texts_all
			
def get_words(sentences):
	words = []
	for sen in sentences:
		tokens = sen.split()
		for token in tokens:
			if token not in words: words.append(token)
	print(len(words))
	return words
 
# data_path = './data/'
data_path = 'alltext.txt'
embed_en_path = 'vec-aen.txt'
embed_tl_path = 'vec-atl.txt'
eos = "<EOS>"; sos = "<SOS>"

input_texts, target_texts = input2target(data_path, sos, eos)
# input_texts, target_texts = get_all_files(data_path, sos, eos)

input_words = get_words(input_texts)
target_words = get_words(target_texts)
if sos in target_words:
	print("Present")

vecs_input = KeyedVectors.load_word2vec_format(embed_tl_path)
vecs_target = KeyedVectors.load_word2vec_format(embed_en_path)
encoder_embed_length = len(vecs_input.get_vector("."))
decoder_embed_length = len(vecs_target.get_vector("."))
max_encoder_seq_length = max([len(words.split()) for words in input_texts])
max_decoder_seq_length = max([len(words.split()) for words in target_texts])

print('Number of samples:', len(input_texts))
print('Length of input word embedding:', encoder_embed_length)
print('Length of output word embedding:', decoder_embed_length)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

encoder_input_data = np.zeros(
	(len(input_texts), max_encoder_seq_length, encoder_embed_length),
		dtype='float32')
decoder_output_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, decoder_embed_length),
		dtype='float32')
decoder_target_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, decoder_embed_length),
		dtype='float32')

print("Transforming text sequences to word embeddings...")
for i, text, in enumerate(input_texts):
	words = text.split()
	for t, word in enumerate(words):
		if word in vecs_input.vocab : encoder_input_data[i, t] = vecs_input.get_vector(word)
		else : 
			print("ENC: Unknown word",word,"found")
			encoder_input_data[i, t] = vecs_input.get_vector("<unk>")
			
for i, text, in enumerate(target_texts):
	words = text.split()
	for t, word in enumerate(words):
		# decoder_target_data is ahead of decoder_output_data by one timestep
		if word in vecs_target : decoder_output_data[i, t] = vecs_target.get_vector(word)
		else : 
			print("DEC: Unknown word",word,"found")
			decoder_output_data[i, t] = vecs_target.get_vector("<unk>")

batch_size = 64  # Batch size for training.
epochs = 100 # Number of epochs to train for.
latent_dim = 256 # Latent dimensionality of the encoding space.				
				
model = Sequential()
model.add(LSTM(latent_dim, input_shape=(None,encoder_embed_length)))
model.add(RepeatVector(max_decoder_seq_length))
model.add(LSTM(latent_dim, return_sequences=True))
model.add(TimeDistributed(Dense(decoder_embed_length, activation='softmax')))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy') # cosine proximity?
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(encoder_input_data, decoder_output_data, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint])
