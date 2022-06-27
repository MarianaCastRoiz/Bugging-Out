import os
import cv2
import pathlib
import matplotlib.pyplot as plt
import numpy as np
def main():
    "Explore the Spider train dataset "
    train_data_dir = './Data/train'
    test_data_dir = './Data/test'
    validate_data_dir = './Data/valid'

    train_data_dir = pathlib.Path(train_data_dir)
    test_data_dir = pathlib.Path(test_data_dir)
    validate_data_dir = pathlib.Path(validate_data_dir)

    train_image_count = get_image_count(train_data_dir)
    test_image_count = get_image_count(test_data_dir)
    validate_image_count = get_image_count(validate_data_dir)

    images_count_list = [train_image_count, test_image_count, validate_image_count]
    print(train_image_count)
    print(test_image_count)
    print(validate_image_count)

    species_list = get_species(train_data_dir)

    show_sample_species(species_list)

    show_dataset_split(images_count_list)

def show_dataset_split(images_count_list):
    "Plots the dataset split"
    x = ['Train', 'Test', 'Validate']
    x_axis = np.arange(len(x))
    plt.xticks(x_axis, x)
    plt.xlabel("Dataset")
    plt.ylabel("Number of Images")
    plt.title("Number of Images in each dataset")
    plt.bar(x_axis,images_count_list,label="Dataset Split")

    plt.savefig('dataset_split.png')

def show_sample_species(species_list):
    "Shows a sample image from each species"
    img = {}
    plt.figure(figsize=(len(species_list),len(species_list)))
    i = 0
    for species in species_list:
        i = i+1
        plt.subplot(5,5,i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        dir = 'Data/test/%s/1.jpg' % (str(species))
        img[species] = cv2.cvtColor(cv2.imread(dir), cv2.COLOR_BGR2RGB)
        plt.imshow(img[species])
        plt.xlabel(species)

    plt.savefig('Spider_grid.png')
    plt.close()


def get_image_count(data_dir):
    "Gets the number of images in the data directory"
    image_count = len(list(data_dir.glob('*/*.jpg')))
    return image_count

def get_species(data_dir):
    "Gets the folder names of the Data directory"
    species_list = os.listdir(data_dir)
    return species_list

if __name__ == '__main__':
    main()

