#Import TensorFlow Keras Libary to create a CNN (Convolutional Neural Network)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D


#TODO: Separate the individual params for easier model modification
class ModelBuilder:
    
    def create_model():
        model = Sequential()

        # Filter = (int) The number of features to extract from the image. Auto detects the features.
        # Kernel Size = (int) Size of the filter matrix
        # Padding = (str) "Same" means to keep the output image size the same as the input, and also keeps the corner portions of the image
        # Activation = (str) Introduces non-linearity to the activation formula
        # Input Shape = (int, int, int) The first two dimensions are the size of the image, the third value is the 3 because it's RGB 
        # 1st convolution layer
        model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))

        # Pooling lowers the size of the feature map and lowers computational cost because it means that we're only keeping features instead of cruft
        model.add(MaxPooling2D(pool_size=2))

        #2nd convolution layer
        model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
        model.add(MaxPooling2D(pool_size=2))

        #3rd convolution layer
        model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))

        #Turns it into a 1D layer after pooling
        model.add(Flatten())

        model.add(Dense(500,activation="relu"))

        #Builds the output layer
        model.add(Dense(2,activation="softmax"))

        return model
    
    def compile_model(model):
        model.compile(loss="categorical_crossentropy" ,optimizer="adam", metrics=["accuracy"])

        return model

    def fit_model(model):
        odel.fit_generator(training_set,validation_data=test_set,epochs=50, steps_per_epoch=len(training_set), validation_steps=len(test_set) )