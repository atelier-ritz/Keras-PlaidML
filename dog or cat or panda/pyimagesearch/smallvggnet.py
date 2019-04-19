# VGGNet-like models share two common characteristics:

# 1. Only 3×3 convolutions are used
# 2. onvolution layers are stacked on top of each other deeper in the network
# architecture prior to applying a destructive pooling operation

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu")) # Rectified Linear Unit
		model.add(BatchNormalization(axis=chanDim))
		# Batch Normalization is used	to normalize the activations
		# of a given input volume before passing it to the next
		# layer in the network.It has been proven	to be very effective
		# at reducing the number of epochs required to train a CNN as
		# well as stabilizing training itself.
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# POOL layers have a primary function of progressively reducing
		# the spatial size (i.e. width and height) of the input volume
		# to a layer. It is common to insert POOL layers between
		# consecutive CONV layers in a CNN architecture.
		model.add(Dropout(0.25))
		# Dropout is an interesting concept not to be overlooked.
		# In an effort to force the network to be more robust we can
		# apply dropout, the process of disconnecting random neurons
		# between layers. This process is proven to reduce overfitting,
		# increase accuracy, and allow our network to generalize better
		# for unfamiliar images. As denoted by the parameter, 25% of the
		# node connections are randomly disconnected (dropped out)
		# between layers during each training iteration.

		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model