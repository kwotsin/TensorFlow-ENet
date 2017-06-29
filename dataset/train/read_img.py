from scipy.misc import imread
import os

files = [file for file in os.listdir('.') if file.endswith('.png')]

for file in files:
	image = imread(file)
	if image.shape[2] != 3:
		print image.shape[2]