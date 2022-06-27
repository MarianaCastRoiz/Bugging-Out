from email.mime import image
import glob
import cv2
from cv2 import resize
import matplotlib.pyplot as plt
import os
from numpy import histogram
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from ModelBuilder import ModelBuilder
import argparse
from tensorflow import keras
 
parser = argparse.ArgumentParser()
parser.add_argument("-t","--Train", help="Show Output")
parser.add_argument("-v","--Visualize", help="Show Output")
parser.add_argument("-lm", "--Load_Model", help="Show Output")
# parser.add_argument("-d", "--Display", help="Show Output")
parser.add_argument("-s", "--Save_Model", help="Show Output")
parser.add_argument("-i", "--ImagePath", help="Show Output")
parser.add_argument("-m", "--ModelPath", help="Show Output")
args = parser.parse_args()

#TODO: Actually split up the program or present users with options
def main():
    print('Setting up train and test sets')

    # train_images,train_labels,test_images,test_labels, valid_images, valid_labels, spiders_label = create_train_test()
    train_ds,test_ds, valid_ds = prepare_data()
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    class_names = train_ds.class_names
    num_classes = len(class_names)

    if args.Train:
        print('Creating logger for tensorboard')
        tensorboard_cb = tensor_board_config()

        print('Create CNN Model')
        model = ModelBuilder.create_model(num_classes)
        model = ModelBuilder.compile_model(model)

        history = ModelBuilder.fit_model(train_ds,valid_ds, tensorboard_cb, model)
        print('Display Model History')
        ModelBuilder.visualize_model(history.history)
        ModelBuilder.save_model(model)
    if args.Save_Model:
        print('Saving Model')
        ModelBuilder.save_model(model)
        ModelBuilder.pickle_history(history)

    if args.Load_Model:
        history = ModelBuilder.load_history()

    if args.ImagePath and args.ModelPath:

        model = ModelBuilder.load_model(args.ModelPath)

        ModelBuilder.predict_image(args.ImagePath,model,class_names)

def tensor_board_config():
    import os
    root_logdir = os.path.join(os.curdir, "model_logs")
    run_logdir = get_run_logdir(root_logdir)

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir,histogram_freq=1)
    return tensorboard_cb

def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%s")
    return os.path.join(root_logdir, run_id)

def prepare_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/train",
    seed=123,
    image_size=(224, 224),
    batch_size=100)

    test_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/test",
    seed=123,
    image_size=(224, 224),
    batch_size=10)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/valid",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=10)

    return train_ds,test_ds, valid_ds

if __name__ == '__main__':
    main()