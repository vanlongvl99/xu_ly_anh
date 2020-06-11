import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import pickle


train =pickle.load(open("/home/vanlong/vanlong/ky6/XuLyAnh/Traffic-Sign-Recognition/GUI/data/train.p", mode="rb"))
valid =pickle.load(open("/home/vanlong/vanlong/ky6/XuLyAnh/Traffic-Sign-Recognition/GUI/data/valid.p", mode="rb"))
test = pickle.load(open("/home/vanlong/vanlong/ky6/XuLyAnh/Traffic-Sign-Recognition/GUI/data/test.p", mode="rb"))
trainX =train["features"]
trainY= train["labels"]
classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}


from sklearn.utils import shuffle
trainX , trainY =shuffle(trainX, trainY)

validX = valid["features"]
validY = valid["labels"]
testX = test["features"]
testY = test["labels"]
trainX = trainX.astype("float")/255.0
validX = validX.astype("float")/255.0
testX = testX.astype("float")/255.0

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
validY = lb.fit_transform(validY)

#Splitting training and testing dataset
print(trainX.shape, validX.shape, trainY.shape, validY.shape)
#Converting the labels into one hot encoding
#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=trainX.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(trainX, trainY, batch_size=32, epochs=epochs, validation_data=(validX, validY))
model.save("my_model.h5")
model.save_weights("weights_model.h5")

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score
validY = pd.read_csv('Test.csv')
labels = validY["ClassId"].values
imgs = validY["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
validX=np.array(data)
pred = model.predict_classes(validX)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
