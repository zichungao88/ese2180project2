import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import mnist_data_loader # class from mnist_data_loader.py

# 1 TODO: Download & load MNIST training & testing datsets

# Set file paths
input_path = './'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

# Load given MNIST dataset
mnist_dataloader = mnist_data_loader.mnist_data_loader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Display loaded images
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize = (30, 20))
    index = 1    
    for i in zip(images, title_texts):        
        image = i[0]        
        title_text = i[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap = plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

images_2_show = []
titles_2_show = []
for i in range(10):
    r = random.randint(1, 5000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(5):
    r = random.randint(1, 5000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

# show_images(images_2_show, titles_2_show)


# 2 TODO: Identify row & column indices of pixels with nonzero intensities




# 3 TODO: Construct matrix A & vector y, solve the least square, & plot values of entries of theta




# 4 TODO: Load images & calculate error rate, false positive rate, & false negative rate of classifier




# 5 TODO: Repeat steps 1-4 w/ only the 1st 100 images




# 6 TODO: Changing the feature set




# 7 TODO: Calculate error rate, false positive, & false negative rate for the new classifier