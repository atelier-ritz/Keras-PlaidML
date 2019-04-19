# =============================================================================
#  train_simple_nn.py
#  ----------------------------------------------------------------------------
#  Source: https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
#  ----------------------------------------------------------------------------
#  Version
#  1.0.0 2019/04/18 First version
#  ----------------------------------------------------------------------------
#  [GitHub] : https://github.com/atelier-ritz
#  ----------------------------------------------------------------------------
#  Animal recognition using Neural Networks
#  Feamework: Keras (wrapper for Tensorflow)
#  Backend: PlaidML (AMD GPU support)
# =============================================================================
#  USAGE
#  python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png


# =====================================
# use plaidml backend (AMD GPU support)
# =====================================
import plaidml.keras
plaidml.keras.install_backend()

# =====================================
# set the matplotlib backend so figures can be saved in the background
# =====================================
import matplotlib
matplotlib.use("Agg")

# =====================================
# import modules
# =====================================
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# =====================================
# construct the argument parser and parse the arguments
# =====================================
'''
args = {
	'dataset': 'animals',
	'model': 'output/simple_nn.model',
	'label_bin': 'output/simple_nn_lb.pickle', 
	'plot': 'output/simple_nn_plot.png'
	}
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


# =====================================
# user-defined area
# =====================================
INIT_LR = 0.01
EPOCHS = 75


# =====================================
# initialize the data and labels
# =====================================
print("[INFO] loading images...")
data = []
labels = []


# =====================================
# grab the image paths and randomly shuffle them
# =====================================
"""
imagePaths is an array contains all the image files
e.g. ['animals\cats\cats_00001.jpg', 'animals\cats\cats_00002.jpg', ...]
"""
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42) # make it reproducible later
random.shuffle(imagePaths)

# loop over the input images
# load the image, resize the image to be 32x32 pixels (ignoring aspect ratio)
# flatten the image into 32x32x3=3072 pixel image into a list, and store the image in the data list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
data = np.array(data, dtype="float") / 255.0 # normalize data [0, 1]
labels = np.array(labels)
# np.save('data.npy', data)
# np.save('labels.npy', labels)
# data = np.load('data.npy')
# labels = np.load('labels.npy')

# partition the data - 75% training, 25% testing
# trainX and testX are already unrolled. It is a 2250 by 3072 matrix
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.25, random_state=42)
# TrainY and testY are list of strings. We need to convert it to one-hot vectors.
# Here, scikit-learn's LabelBinarizer is used, but for two-class binary classification
# you should use Keras' to_categorical() function
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY) # this line determines the classes + encodes labels
testY = lb.transform(testY) # no other labels are introduced, so just use transform() to encode the labels
# print(lb.classes_) # returns the correspondence

# define the 3072-1024-512-3 NN
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# compile the model using SGD as our optimizer and categorical cross-entropy loss.
# Categorical cross-entropy is used as the loss for nearly all networks trained to perform classification.
# However, for 2-class classification, you'll want to use binary_crossentropy
print("[INFO] training network...")
opt = SGD(lr=INIT_LR) # Stochastic Gradient Descent
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))


# Save the model and labelBinarizer to disk. It includes:
# - the architecture of the model, allowing to re-create the model
# - the weights of the model
# - the training configuration (loss, optimizer)
# - the state of the optimizer, allowing to resume training exactly where you left off.
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb") # write in binary mode
f.write(pickle.dumps(lb))
f.close()