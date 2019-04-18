# =============================================================================
#  MNIST.js
#  ----------------------------------------------------------------------------
#  (C)2019 Atelier_Ritz
#  This software is released under the MIT License.
#  http://opensource.org/licenses/mit-license.php
#  ----------------------------------------------------------------------------
#  Version
#  1.1.0 2019/04/18 Added function to plot learning curves
#  1.0.0 2019/04/18 First version
#  ----------------------------------------------------------------------------
#  [GitHub] : https://github.com/atelier-ritz
#  ----------------------------------------------------------------------------
#  Handwritten digit recognition using Neural Networks
#  Feamework: Keras (wrapper for Tensorflow)
#  Backend: PlaidML (AMD GPU support)
# =============================================================================

# =====================================
# use plaidml backend (AMD GPU support)
# =====================================
import plaidml.keras
plaidml.keras.install_backend()

# =====================================
# import modules
# =====================================
import os
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import matplotlib
import matplotlib.pyplot as plt
from statistics import mean

# =====================================
# user-defined area
# =====================================
FILENAME_MODEL = 'cnn_model.json'
FILENAME_WEIGHT = 'cnn_model_weights.hdf5'
NUM_LABELS = 10 # in this example, we will have digits 0-9

# =====================================
# body
# =====================================
def getFileModel():
	return os.path.join(os.getcwd(), FILENAME_MODEL)

def getFileWeight():
	return os.path.join(os.getcwd(), FILENAME_WEIGHT)

def saveModel(model):
	json_string = model.to_json()
	open(getFileModel(), 'w').write(json_string)
	model.save_weights(getFileWeight())
	print('Model and weight have been saved.')

def loadModel():
	json_string = open(getFileModel()).read()
	model = model_from_json(json_string)
	model.load_weights(getFileWeight())
	print('Model and weights have been loaded.')
	model.summary()
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) # specify and cost function
	return model

def plotData(X_sample,y_sample,displayStartIndex=0):
	plt.figure()
	for i in range(9):
		plt.subplot(3, 3, i + 1)
		plt.tight_layout()
		plt.imshow(X_sample[i+displayStartIndex], cmap='gray', interpolation='none')
		plt.title("[#{}]-{}".format(i + displayStartIndex,y_sample[i+displayStartIndex]))
		plt.xticks([])
		plt.yticks([])
	plt.show()

def plotLearningCurve(history):
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(history.history['acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='lower right')

	plt.subplot(2, 1, 2)
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper right')

	plt.tight_layout()
	plt.show()

def getData():
	"""
	x_train, x_test: uint8 array of	grayscale image	data with shape(num_samples, 28, 28).
	y_train, y_test: uint8 array of	digit labels(integers in range 0-9) with shape(num_samples,).
	Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
	"""
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	return x_train, x_test, y_train, y_test

def preprocessX(mat):
	"""
	unroll input matrices
	:param mat: ndarray, uint8, shape=(num_samples, num_row, num_col)
	:return: ndarray, float32, shape=(num_samples, num_row * num_col)
	"""
	num_samples, num_row, num_col = mat.shape
	vect = mat.reshape(num_samples, num_row * num_col)
	vect_normalized = vect.astype('float32') / 255
	return vect_normalized

def createModel():
	"""
	input (1st layer): 784 nodes
	2nd layer: 64 nodes
	output (3rd layer): 10 nodes
	"""
	model = Sequential()
	model.add(Dense(64, activation='relu', input_dim=784)) # 2nd layer
	model.add(Dense(10, activation='softmax')) # 3rd layer
	model.summary()
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) # specify and cost function
	return model

def trainModel(model, x_train, y_train, batch_size=100,	epochs=12, verbose=1):
	# training
	# batch_size: Integer. Number of samples per propagation. The higher the more memory size you will need. (default 32)
	# epochs: Integer. Number of iteration over the entire x and y data provided. Too high -> overfitting to train data
	# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
	x_train = preprocessX(x_train)
	y_train = to_categorical(y_train, NUM_LABELS) # convert to a 60000 by 10 logical matrix
	history = model.fit(x_train, y_train,	batch_size,	epochs, verbose)
	plotLearningCurve(history)
	return model

def evaluateModel(model, x_test, y_test):
	x_test = preprocessX(x_test)
	y_test = to_categorical(y_test, NUM_LABELS) # convert to a 60000 by 10 logical matrix
	score = model.evaluate(x_test, y_test)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	# plotLearningCurve(score.)

def predict(model,x_predict, startIndex=-1):
	x_predict_vect = preprocessX(x_predict)
	y_predict = model.predict(x_predict_vect).argmax(axis=-1)
	if startIndex >= 0:
		print(y_predict[startIndex:startIndex+9])
		plotData(x_predict, y_predict, startIndex)

if __name__ == "__main__":
	x_train, x_test, y_train, y_test = getData()

	# if you don't have a trained model, run this
	plotData(x_train, y_train, displayStartIndex=0)
	model = createModel()
	model = trainModel(model, x_train, y_train, batch_size=100,	epochs=12, verbose=1)
	saveModel(model)

	# if you already have a model, run this
	# model = loadModel()
	# evaluateModel(model, x_test, y_test)
	# predict(model,x_test, 10) # here, we try to predict the test samples