import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import mnist_data_loader # class from mnist_data_loader.py

# 1 TODO: Download & load MNIST training & testing datsets
# DONE
# Set file paths
input_path = './'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

# Load given MNIST dataset
mnist_dataloader = mnist_data_loader.mnist_data_loader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# 2 TODO: Identify row & column indices of pixels with nonzero intensities
# DONE
# pixel intensity: black = 0, 255 = white
image_quantity = len(x_train) # truncated to 5000 out of 60000
pixel_dimensions = (28, 28) # 28x28 pixels

nonzero_intensities = []
for i in range(600):
    for j in range(pixel_dimensions[0]):
        for k in range(pixel_dimensions[1]):
            if x_train[i][j][k] != 0 and (j, k) not in nonzero_intensities:
                nonzero_intensities.append((j, k))
feature_quantity = len(nonzero_intensities) # number of features used for classification


# 3 TODO: Construct matrix A & vector y, solve the least square, & plot values of entries of theta




# 4 TODO: Load images & calculate error rate, false positive rate, & false negative rate of classifier




# 5 TODO: Repeat steps 1-4 w/ only the 1st 100 images




# 6 TODO: Changing the feature set




# 7 TODO: Calculate error rate, false positive, & false negative rate for the new classifier