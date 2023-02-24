from keras.models import load_model
import numpy as np
import sys
import os
from PIL import Image
import tensorflow as tf


np.set_printoptions(threshold=np.inf)
data = []
model = load_model('traffic_classifier.h5')

try:
	image = Image.open('image.jpg')
	image = image.resize((30,30))
	image = np.array(image)
	data.append(image)
	X_test=np.array(data)
	pred = model.predict_classes(X_test)
	print(pred)
except:
	print("Error loading image")

