# *******************************************************************
# *  It evaluates the model's performance and then plots            *
# *  the weights of each convolutional layer                        *
# *                                                                 *
# *                                                                 *
# *                                                                 *
# *                                                                 *
# * Author: Mahmoud Jadaan                                          *
# *******************************************************************

import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K


def get_weights(model):
    """
    It plots the weights of the convolutional layers for this kind of architecture repetitive (Conv -> Pool)* n
    :param model: the desired model
    :return: figures of weights
    """
    config = {}
    weight = {}
    i = 0
    for layer in model.layers:
        config[i] = layer.get_config()
        weight[i] = layer.get_weights()
        i = i + 1
    filter_dim = config[0]['kernel_size']
    for i in range(len(model.layers)):
        if 'conv2d' in config[i]['name']:
            if i == 0:
                depth = 1
            else:
                depth = config[(i - 2)]['filters']
            weight[i][0] = weight[i][0].reshape(config[i]['filters'], depth, filter_dim[0], filter_dim[0])
            for j in range(config[i]['filters']):
                plt.figure(1+i, figsize=(8, 8))
                plt.subplot(4, 4, j+1)
                plt.imshow(weight[i][0][j][0].reshape(filter_dim[0], filter_dim[0]), cmap='jet', interpolation='none')
                plt.title("Filter {}".format(j+1))
            plt.show()


# from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input
    model_multi_inputs_cond = True

    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


# from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def display_activations(activation_maps, model):
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    nr_rows = 4
    nr_cols = 3
    for i, activation_map in enumerate(activation_maps):

        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.vstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=1)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.figure(1)
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.title(model.get_layer(activation_map, i))
        plt.show()


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


# **************************************
# * Load MINST Data
# **************************************
(X_train_original, Y_train_original), (X_test_original, Y_test_original) = mnist.load_data()
Train_Image_Number, Test_Image_Number, Image_Width,\
 Image_Height = image_properties(X_train_original, X_test_original)

# **************************************
# * Reshaping the original X data
# **************************************
X_test_original = X_test_original.reshape(Test_Image_Number, Image_Width, Image_Height, 1)

Nr_Classes = 10
Y_test_original = create_label(Y_test_original, Nr_Classes)
print("The X test MNIST data:")
print("  X_Test_original shape:", X_test_original.shape)
print("  Y_Test_Labels shape:", Y_test_original.shape)

# *****************************************
# *  Load a trained model and its weights *
# *****************************************
model = load_model("Trained_Model/Model_Original_Traces.hdf5")
model.load_weights("Trained_Model/Best_weights_Original_Traces.hdf5")

# ***********************************
# *   Evaluate model on test data   *
# ***********************************
batchSize = 200
loss, accuracy = model.evaluate(X_test_original, Y_test_original, batchSize, verbose=0)
print('The Accuracy of Model: {:.2f}%'.format(accuracy * 100))
print('The loss on test data: {:.2f}'.format(loss))


# ***************************************
# *     Plot activation maps            *
# ***************************************
get_weights(model)
activation_map = get_activations(model, X_test_original[0:1], print_shape_only=True)  # with just one sample.
display_activations(activation_map, model)

