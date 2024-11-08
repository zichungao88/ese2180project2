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

# load given MNIST dataset & truncate to the 1st 5000 images
mnist_dataloader = mnist_data_loader.mnist_data_loader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = x_train[:5000]
x_test = x_test[:5000]
y_train = y_train[:5000]
y_test = y_test[:5000]

# 2 TODO: Identify row & column indices of pixels with nonzero intensities
# DONE
# pixel intensity: black = 0, 255 = white
image_quantity = len(x_train) # truncated to 5000 out of 60000
pixel_dimensions = (28, 28) # 28 x 28 pixels
nonzero_intensities = []

def calculate_feature(intensities, dimensions, image_number):
    for i in range(image_number): # 600 <= x <= 5000
        for j in range(dimensions[0]):
            for k in range(dimensions[1]):
                if x_train[i][j][k] != 0 and (j, k) not in intensities:
                    intensities.append((j, k))
    feature_quantity = len(intensities) # number of features used for classification
    return feature_quantity

feature_quantity = calculate_feature(nonzero_intensities, pixel_dimensions, 1000)
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
# IN PROGRESS (ALMOST DONE)
# reconstruct A & y (only theta remains unchanged for testing)
A1, y1 = construct_A_y(x_test, y_test, nonzero_intensities, image_quantity, feature_quantity)

# calculate f_tilde(x) i.e. least squares classifier
def calculate_classifier(A_matrix, theta_vector):
    least_squares = np.matmul(A_matrix, theta_vector)
    # print(len(least_squares))

    classifier = np.sign(least_squares)
    # print(len(classifier))

    return classifier

least_squares_classifier = calculate_classifier(A1, theta)

# for i in range(image_quantity):
#     print((least_squares_classifier[i], y1[i]))

# compare w/ test data
def calculate_error(classifier, actual, image_number):
    total_positive = 0
    total_negative = 0
    total_error = 0
    total_false_positive = 0
    total_false_negative = 0
    for i in range(image_number):
        if actual[i] == 0: # class 1
            total_positive += 1
            if classifier[i] == -1: # false negative
                total_error += 1
                total_false_negative += 1
        else: # class -1
            total_negative += 1
            if classifier[i] == 1: # false positive
                total_error += 1
                total_false_positive += 1
    
    error_rate = total_error / image_number
    false_positive_rate = total_false_positive / total_negative
    false_negative_rate = total_false_negative / total_positive

    errors = [error_rate, false_positive_rate, false_negative_rate]
    return errors

error_list = calculate_error(least_squares_classifier, y1, image_quantity)
error_rate = format(error_list[0], '.0%')
false_positive_rate = format(error_list[1], '.0%')
false_negative_rate = format(error_list[2], '.0%')
print("For 5,000 images:")
print("Error Rate: " + error_rate)
print("False Positive Rate: " + false_positive_rate)
print("False Negative Rate: " + false_negative_rate)


# 5 TODO: Repeat steps 1-4 w/ only the 1st 100 images
# IN PROGRESS (gotta finish # 4 first) (bye)
x_train100 = x_train[:100]
x_test100 = x_test[:100]
y_train100 = y_train[:100]
y_test100 = y_test[:100]

image_quantity100 = len(x_train100)
nonzero_intensities100 = []
feature_quantity100 = calculate_feature(nonzero_intensities100, pixel_dimensions, 100) # caps @ 100

A100, y100 = construct_A_y(x_train100, y_train100, nonzero_intensities100, image_quantity100, feature_quantity100)
theta100 = np.linalg.lstsq(A100, y100, rcond=None)[0]

theta_plot100 = np.zeros(pixel_dimensions)
for i, (j, k) in enumerate(nonzero_intensities100):
    theta_plot100[j, k] = theta100[i]
plt.figure() # avoid overriding previous figure
plt.imshow(theta_plot100)
plt.colorbar(label="Value of θ")
plt.title("Entries of θ at Different Pixel Coordinates")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.savefig('theta100.png')

A1001, y1001 = construct_A_y(x_test100, y_test100, nonzero_intensities100, image_quantity100, feature_quantity100)
least_squares_classifier100 = calculate_classifier(A1001, theta100)

error_list100 = calculate_error(least_squares_classifier100, y1001, image_quantity100)
error_rate100 = format(error_list100[0], '.0%')
false_positive_rate100 = format(error_list100[1], '.0%')
false_negative_rate100 = format(error_list100[2], '.0%')
print("\nFor 100 images:")
print("Error Rate: " + error_rate100)
print("False Positive Rate: " + false_positive_rate100)
print("False Negative Rate: " + false_negative_rate100)


# 6 TODO: Changing the feature set
# CONTINUE NEXT (again, gotta finish # 4 first)



# 7 TODO: Calculate error rate, false positive, & false negative rate for the new classifier