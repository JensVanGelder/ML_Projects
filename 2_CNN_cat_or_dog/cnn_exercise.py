# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:17:31 2017

@author: jens
"""

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation ='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation ='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden layer
classifier.add(Dense(units=128, activation = 'relu'))
# Output layer
classifier.add(Dense(units=1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images, skip this step if you already have your classifier
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=200,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=100)


# serialize model to JSON
classifier_json = classifier.to_json()
with open("dataset/classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")
 
# Part 3 - Loading classifier and making predictions
 
# load json and create model
json_file = open('dataset/classifier.json', 'r')
classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(classifier_json)
# load weights into new model
loaded_classifier.load_weights("classifier.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_classifier.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Make predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
