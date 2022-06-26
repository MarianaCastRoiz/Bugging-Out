#Import TensorFlow Keras Libary to create a CNN (Convolutional Neural Network)
from gc import callbacks
import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
import time
import pickle


#TODO: Separate the individual params for easier model modification
class ModelBuilder:
    
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(15, activation=tf.nn.softmax)
        ])

        return model
    
    def compile_model(model):
        model.compile(loss="sparse_categorical_crossentropy" ,optimizer="adam", metrics=["accuracy"])

        return model

    def fit_model(model, training_set, test_set, training_labels, tensorboard_cb):
        history = model.fit(training_set, training_labels, batch_size=128, epochs=15, validation_split=0.2, callbacks=[tensorboard_cb])

        return history
    def pickle_history(history):
        #Save the history
        with open('src/history', 'wb') as file:
            pickle.dump(history.history, file)

    def save_model(model):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model.save('models/model-%s',timestr)

    def load_history():
        history = pickle.load(open('src/history.json', "rb"))
        return history

    def visualize_model(history):
        figure = plt.figure(figsize=(10,5))

        plt.subplot(221)
        plt.plot(history['accuracy'],'bo--', label="acc")
        plt.plot(history['val_accuracy'], 'ro--', label="val_access")
        plt.title("train_acc vs val_acc")
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.legend()

        #Loss Function
        plt.subplot(222)
        plt.plot(history['loss'],'bo--', label="loss")
        plt.plot(history['val_loss'], 'ro--', label="val_loss")
        plt.title("train_loss vs val_loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend()

        plt.savefig('model-plot.png',format='png')