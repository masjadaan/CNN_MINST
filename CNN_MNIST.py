# *******************************************************************
# *  It creates a simple CNN model and runs it against              *
# *  the augmented images                                           *
# *                                                                 *
# *                                                                 *
# *                                                                 *
# *                                                                 *
# * Author: Mahmoud Jadaan                                          *
# *******************************************************************

import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import time


def divide_data(x_data, y_data, upper_limit, lower_limit):
    """
    It divides the data set into sections
    :param float x_data  : n-dimensional array contains the data.
    :param int y_data    : Labels of the data.
    :param int upper_limit: Data upper limit.
    :param int lower_limit: Data lower limit.
    :return: Divided data and its labels.
    """
    x_train = x_data[lower_limit: upper_limit]
    y_train = y_data[lower_limit: upper_limit]
    return x_train, y_train


def data_limit(x_train, ratio):
    """
    It limits the dataset, usually 20% for test and validation
    :param float x_train: n-dimensional array contains the data.
    :param float ratio  : ratio of the original data.
    """
    x = int(len(x_train) * ratio)
    return x


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def architecture(input_shape, filter_shape, number_classes):
    """
    It defines the architecture of the model, (C --> P)*2 --> FC.
    Note: the number of classes is defined as 10 classes, but can be modified as required
    :param int input_shape  : Width, height and depth of the input data.
    :param int filter_shape : Width, height of the filter.
    :param int number_classes   : desired number of classes.
    """
    model = Sequential()
    activation = "relu"
    # ----------------------------------------------------------------------------------------------------------------------

    block_1_filters = 8
    # Conv
    model.add(Conv2D(block_1_filters, kernel_size=filter_shape, padding='same',
                     data_format="channels_last", activation=activation, use_bias=True,
                     input_shape=input_shape))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------
    block_2_filters = 16
    # Conv
    model.add(Conv2D(block_2_filters, kernel_size=filter_shape, padding='same',
                     data_format="channels_last", activation=activation, use_bias=True))
    # ----------------------------------------------------------------------------------------------------------------------

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    # ----------------------------------------------------------------------------------------------------------------------
    # FC
    final_dense_output = number_classes
    model.add(Flatten())
    model.add(Dense(32, activation=activation))
    model.add(Dense(final_dense_output, activation='softmax'))
    return model


def train_model(x, y, x_val, y_val, loss, epochs, batch):
    """
    :param float x: n dimensional array holds training data.
    :param int y: n dimensional array holds labels data.
    :param float x_val: n dimensional array holds validation data.
    :param int y_val: n dimensional array holds label data.
    :param loss: the desired loss function.
    :param int epochs: number of iteration throughout the training data.
    :param int batch: size of a batch.
    :return: trained model
    """
    adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    time_callback = TimeHistory()
    # checkpoint
    # 1- store the best weights
    file_path = "Trained_Model/Best_weights_Original_Traces.hdf5"
    best_accuracy_check_point = ModelCheckpoint(file_path, monitor='val_acc',
                                                verbose=1, save_best_only=True, mode='max')

    # 2- Early Stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    callbacks_list = [best_accuracy_check_point, time_callback, early_stop]

    model.fit(x, y, batch, epochs, verbose=1, validation_data=(x_val, y_val), shuffle=1, callbacks=callbacks_list)

    time_per_epoch = time_callback.times
    # sum up the time of all epochs , divide by 60 to convert to min
    t = np.sum(time_per_epoch)/60
    print(t)
    return model


# **************************************
# * read the Processed MNIST
# **************************************
X_file = h5py.File("Processed_MNIST/X_Processed_MNIST.hdf5", 'r')  # read
X_train_original = np.array(X_file.get('X_P_MNIST'))

Y_file = h5py.File("Processed_MNIST/Y_Processed_MNIST.hdf5", 'r')  # read
Y_train_original = np.array(Y_file.get('Y_P_MNIST'))
print("X_Processed_MNIST shape:", X_train_original.shape)
print("Y_Processed_MNIST shape:", Y_train_original.shape)
print('********************************\n')


# ************************************************************
# * Divide training MNIST data into validation and training
# ************************************************************
# 1- Validation data range:
valDataUpperLimit = data_limit(X_train_original, 0.2)
valDataLowerLimit = 0
x_Validation, y_Validation = divide_data(X_train_original, Y_train_original,
                                         valDataUpperLimit, valDataLowerLimit)

# 2- training data range:
trainDataUpperLimit = data_limit(X_train_original, 1)
trainDataLowerLimit = valDataUpperLimit
x_Train, y_Train = divide_data(X_train_original, Y_train_original,
                               trainDataUpperLimit, trainDataLowerLimit)

print("The X_Data:")
print("  x_Train      :", x_Train.shape)
print("  x_Validation :", x_Validation.shape)
print("-----------------------------------------------------------")

print("The Y_Data:")
print("  y_Train      :", y_Train.shape)
print("  y_Validation :", y_Validation.shape)


# ************************************
# *     Input Dimensions
# ************************************
Input_Shape = (len(x_Train[0]), len(x_Train[1]), 1)

# ************************************
# *     Filter Dimensions
# ************************************
Filter_Shape = (11, 11)

NrClasses = 10
model = architecture(Input_Shape, Filter_Shape, NrClasses)

# summarize the architecture
model.summary()


# *********************************
# *     Train model               *
# *********************************
Loss_Function = 'mean_squared_error'  # categorical_crossentropy
Batch_Size = 200
Epochs = 20
model = train_model(x_Train, y_Train, x_Validation, y_Validation, Loss_Function, Epochs, Batch_Size)


# ***********************************
# *   Save the model                *
# ***********************************
model.save("Trained_Model/Model_Original_Traces.hdf5")

# ***********************************
# *   Close all Open Files
# ***********************************
X_file.close()
Y_file.close()

