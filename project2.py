import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import mnist_data_loader # class from mnist_data_loader.py

# 1 TODO: Download & load MNIST training & testing datsets
# DONE
# set file paths
input_path = './'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

# load given MNIST dataset
mnist_dataloader = mnist_data_loader.mnist_data_loader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# 2 TODO: Identify row & column indices of pixels with nonzero intensities
# DONE
# pixel intensity: black = 0, 255 = white
image_quantity = len(x_train) # truncated to 5000 out of 60000
pixel_dimensions = (28, 28) # 28 x 28 pixels

nonzero_intensities = []
for i in range(1000): # 600 <= x <= 5000
    for j in range(pixel_dimensions[0]):
        for k in range(pixel_dimensions[1]):
            if x_train[i][j][k] != 0 and (j, k) not in nonzero_intensities:
                nonzero_intensities.append((j, k))
feature_quantity = len(nonzero_intensities) # number of features used for classification
# print(feature_quantity) # should not exceed 28 * 28 = 784


# 3 TODO: Construct matrix A & vector y, solve the least square, & plot values of entries of theta
# DONE
def construct_A_y(x_data, y_data, features, N, M):
    # A -> N x M = image_quantity x feature_quantity
    A = np.zeros((N, M))
    feature_functions = [lambda x, pixel = pixel: x[pixel[0]][pixel[1]] for pixel in features]
    for i in range(N):
        for j in range(M):
            A[i, j] = feature_functions[j](x_data[i])
    return A, y_data

A, y = construct_A_y(x_train, y_train, nonzero_intensities, image_quantity, feature_quantity)

# solve least squares
theta = np.linalg.lstsq(A, y, rcond=None)[0]

# map theta to 28 x 28 grid (will be sparse b/c # of features < # of pixels)
theta_plot = np.zeros(pixel_dimensions)
for i, (j, k) in enumerate(nonzero_intensities):
    theta_plot[j, k] = theta[i]
plt.imshow(theta_plot)
plt.colorbar(label="Value of θ")
plt.title("Entries of θ at Different Pixel Coordinates")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.savefig('theta.png')
# plt.show()


# 4 TODO: Load images & calculate error rate, false positive rate, & false negative rate of classifier




# 5 TODO: Repeat steps 1-4 w/ only the 1st 100 images




# 6 TODO: Changing the feature set




# 7 TODO: Calculate error rate, false positive, & false negative rate for the new classifier