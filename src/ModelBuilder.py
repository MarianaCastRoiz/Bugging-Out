#Import TensorFlow Keras Libary to create a CNN (Convolutional Neural Network)
from gc import callbacks
import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
import time
import pickle

class ModelBuilder:
    
    def create_model(num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
            ])

        return model
    
    def compile_model(model):
        model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        return model

    def fit_model(train_ds, valid_ds, tensorboard_cb, model):
        history = model.fit(train_ds, validation_data=valid_ds,epochs=20, callbacks=[tensorboard_cb])

        return history
    def pickle_history(history):
        #Save the history
        with open('src/history', 'wb') as file:
            pickle.dump(history.history, file)

    def save_model(model):
        dir = 'models/' + time.strftime("model_%Y%m%d-%H%M%S")
        model.save(dir)

    def load_model(modelPath):
        model = tf.keras.models.load_model(modelPath)
        return model
    
    def predict_image(image_path, model, class_names):
        img = tf.keras.utils.load_img(
        image_path, target_size=(224, 224)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

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
        plt.close()