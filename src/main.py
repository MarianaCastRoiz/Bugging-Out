import glob
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from ModelBuilder import ModelBuilder
import argparse
 
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

        # train_images, train_labels = shuffle(train_images,train_labels, random_state=25)
        print('Create CNN Model')
        model = ModelBuilder.create_model()
        model = ModelBuilder.compile_model(model)
        history = ModelBuilder.fit_model(model, train_images, test_images, train_labels)
        viz_history = history

    if args.Save_Model:
        print('Saving Model')
        ModelBuilder.save_model(model)
        ModelBuilder.pickle_history(history)

    if args.Load_Model:
        history = ModelBuilder.load_history()

    if args.Display:
        print('Displaying model history')
        ModelBuilder.visualize_model(viz_history)

def create_train_test():
    spiders = ["Black Widow", "Blue Tarantula", "Bold Jumper", "Brown Grass Spider", "Brown Recluse Spider",
        "Deinopis Spider", "Golden Orb Weaver", "Hobo Spider", "Huntsman Spider", "Ladybird Mimic Spider",
        "Peacock Spider", "Red Knee Tarantula", "Spiny-backed Orb-weaver", "White Kneed Tarantula", "Yellow Garden Spider"]

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