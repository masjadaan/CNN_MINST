# *******************************************************************
# * Pre-processing MNIST images                                     *
# * Each image has been shift randomly to right, left, up and down  *
# * then they have been added onto the original images to form the  *
# * final dataset that contains now 120000                          *
# *                                                                 *
# *                                                                 *
# * Author: Mahmoud Jadaan                                          *
# *******************************************************************

import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import h5py


def image_properties(train_image, test_image):
    """
    It returns the properties of test and train images
    :param float train_image: n-dimensional array holds the training images
    :param float test_image: n-dimensional array holds the test images
    :return: number of train and test images in addition to width and height
    """
    train_nr = len(train_image)
    test_nr = len(test_image)
    image_width = len(train_image[0])
    image_height = len(test_image[1])
    return train_nr, test_nr, image_width, image_height


def create_label(label_vector, nr_classes):
    """
    It creates labels depending on the desired number of classes.
    :param int nr_classes: Desired number of classes.
    :param float label_vector: contains the labels
    """
    y = np_utils.to_categorical(label_vector, nr_classes)
    return y


def normalize(ndarr):
    """
    Takes an n-dimensions array and divides its element by the largest number within the array.
    :param ndarr: n-dimensions array.
    :returns n-dimensional array.
    """
    ndarr = ndarr.astype('float32')
    maxamp = np.ndarray.max(ndarr)
    ndarr /= maxamp
    return ndarr


def plot_img(nr_img, x_img, y_img):
    """
    It plot specified number of traces in one figure
    :param int nr_img: desired number of traces
    :param float x_img: n-dimensional array holds the traces
    :param float y_img: n-dimensional array holds the labels
    """
    for i in range(nr_img):
        plt.figure(1, figsize=(10, 7))
        plt.subplot(nr_img, nr_img/2, i + 1)
        plt.imshow(x_img[i].reshape(len(x_img[0]), len(x_img[1])), cmap='gray')
        plt.title("Class {}".format(y_img.argmax(1)[i]))
        # plt.title("Class {}".format(y_img[i]))
    plt.show()


def data_aug(x_train, y_train, shift_value):
    """
    It uses Keras data augmentation function to generate randomly shifted data with its labels
    :param float x_train: 4 dimensional tensor, its second dimension contains amplitude.
    :param float y_train: 2 dimensional tensor, its second dimension contains the calss number.
    :param float shift_value: the desired shift.
    """
    datagen = ImageDataGenerator(height_shift_range=shift_value)

    i = 0
    for x_train_aug, y_train_aug in datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=False):
        """
        The .flow() generates batches of randomly transformed images and saves the results.
        :param int batch_size : indicate how many samples from X_train we want to use.
        :param boolean shuffle: To shuffle the data each round.
        """
        i += 1
        if i > 10:
            break  # otherwise the generator would loop indefinitely
    return x_train_aug, y_train_aug


# **************************************
# * Load MINST Data
# **************************************
(X_train_original, Y_train_original), (X_test_original, Y_test_original) = mnist.load_data()
Train_Image_Number, Test_Image_Number, Image_Width,\
 Image_Height = image_properties(X_train_original, X_test_original)


# **************************************
# * Reshaping the original X data
# **************************************
X_train_original = X_train_original.reshape(Train_Image_Number, Image_Width, Image_Height, 1)
X_test_original = X_test_original.reshape(Test_Image_Number, Image_Width, Image_Height, 1)

Nr_Classes = 10
Y_train_original = create_label(Y_train_original, Nr_Classes)
print("The X Original MNIST data:")
print("  X_Train_original shape:", X_train_original.shape)
print("  Y_Train_Labels shape:", Y_train_original.shape)


# **************************************
# * Normalize [0, 1]
# **************************************
X_train_original = normalize(X_train_original)


# **************************************
# * Data Augmentation
# **************************************
Shift = 0.2
X_train_Aug, Y_train_Aug = data_aug(X_train_original, Y_train_original, Shift)


# ************************************************
# * Concatenate the original and augmented data
# ************************************************
X_train_concatenated = np.concatenate((X_train_original, X_train_Aug))
Y_train_concatenated = np.concatenate((Y_train_original, Y_train_Aug))
print("The Preprocessed MNIST data:")
print("  X_Train_original shape:", X_train_concatenated.shape)
print("  Y_Train_Labels shape:", Y_train_concatenated.shape)


# ************************************************
# * Storing the final processed MNIST
# ************************************************
X_Processed_File = h5py.File("Processed_MNIST/X_Processed_MNIST.hdf5", "w")
X_Processed_Dataset = X_Processed_File.create_dataset('X_P_MNIST', data=X_train_concatenated)

Y_Processed_File = h5py.File("Processed_MNIST/Y_Processed_MNIST.hdf5", "w")
Y_Processed_Dataset = Y_Processed_File.create_dataset('Y_P_MNIST', data=Y_train_concatenated)

# **************************************
# * Plot n images
# **************************************
Nr_Images = 10
plot_img(Nr_Images, X_train_original, Y_train_original)
plot_img(Nr_Images, X_train_Aug, Y_train_Aug)


# **************************************
# * close all open files
# **************************************
X_Processed_File.close()
Y_Processed_File.close()

