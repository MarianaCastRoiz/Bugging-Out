import glob
import cv2
import matplotlib.pyplot as plt
import os

def main():
    print("Bug Identifying Model")
    read_folder()

def read_folder():
    #Set up the folder names
    train_folder = "../Data/train/Black Widow"
    test_folder = "../Data/test"

    #Read the test folder
    images = [cv2.imread(file) for file in glob.glob("Data/train/*/*.jpg")]
    thing = images[0]
    cv2.imshow("spider",thing)

if __name__ == '__main__':
    main()