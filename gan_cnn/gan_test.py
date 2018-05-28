from __future__ import print_function, division

from keras.models import Model, load_model
import sys
import numpy as np
from scipy.ndimage import filters
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
	
if __name__ == '__main__':

	model = load_model(sys.argv[1])
	end_loop = False
	while not end_loop :
		noise = np.random.normal(0, 1, (1,128))
		img2 = model.predict(noise)
		img2 = np.uint8(img2*127.5 + 127.5)[0]
		imgplot = plt.imshow(img2), plt.show()
		x = input("Type anything to generate another image or q to quit... ")
		if(x == 'q') : end_loop = True