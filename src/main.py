import glob
import cv2
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
parser.add_argument("-d", "--Display", help="Show Output")
parser.add_argument("-s", "--Save_Model", help="Show Output")
args = parser.parse_args()

#TODO: Actually split up the program or present users with options
def main():
    if args.Train:
        print('Setting up train and test sets')
        train_images,train_labels,test_images,test_labels = create_train_test()

        print('Creating logger for tensorboard')
        tensorboard_cb = tensor_board_config()

        # train_images, train_labels = shuffle(train_images,train_labels, random_state=25)
        print('Create CNN Model')
        model = ModelBuilder.create_model()
        model = ModelBuilder.compile_model(model)
        history = ModelBuilder.fit_model(model, train_images, test_images, train_labels, tensorboard_cb)

        print('Display Model History')
        ModelBuilder.visualize_model(history.history)

    if args.Save_Model:
        print('Saving Model')
        ModelBuilder.save_model(model)
        ModelBuilder.pickle_history(history)

    if args.Load_Model:
        history = ModelBuilder.load_history()

    if args.Display:
        print('Displaying model history')
        ModelBuilder.visualize_model(viz_history)

def tensor_board_config():
    import os
    root_logdir = os.path.join(os.curdir, "model_logs")
    run_logdir = get_run_logdir(root_logdir)

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir,histogram_freq=1)
    return tensorboard_cb

def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%s")
    return os.path.join(root_logdir, run_id)


def create_train_test():
    spiders = ["BlackWidow", "BlueTarantula", "BoldJumper", "BrownGrassSpider", "BrownRecluseSpider",
        "DeinopisSpider", "GoldenOrbWeaver", "HoboSpider", "HuntsmanSpider", "LadybirdMimicSpider",
        "PeacockSpider", "RedKneeTarantula", "Spiny-backedOrb-weaver", "WhiteKneedTarantula", "YellowGardenSpider"]

    spiders_label = {spiders: i for i, spiders in enumerate(spiders)}

    train_labels = []
    test_labels = []
    test_images = []
    test_file_name = []
    train_images = []
    train_file_name = []

    #Set up the folder names
    train_folder = "Data/train"
    test_folder = "Data/test"

    #Read the train list with the label list
    train_folders = os.listdir(train_folder)

    for folder in train_folders:
        image_path = os.path.join(train_folder,folder)
        for file in glob.glob(os.path.join(image_path,"*.jpg")):
            train_labels.append(spiders_label[folder])
            train_images.append(cv2.imread(file))
            train_file_name.append(file)

    test_folders = os.listdir(test_folder)
    for folder in test_folders:
        image_path = os.path.join(test_folder,folder)
        for file in glob.glob(os.path.join(image_path,"*.jpg")):
            test_labels.append(spiders_label[folder])
            test_images.append(cv2.imread(file))
            test_file_name.append(file)


    train_images = np.array(train_images, dtype='float32')
    test_images = np.array(train_images, dtype='float32')

    train_labels = np.array(train_labels, dtype='int32')
    test_labels = np.array(test_labels, dtype='int32')
    # cv2.imshow(train_labels[0],train_img[0])
    # cv2.waitKey(0)
    
    return train_images,train_labels,test_images,test_labels


if __name__ == '__main__':
    main()