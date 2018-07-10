
This project uses a simple CNN against MNIST dataset; the idea behind the project is to use a data augmentation technique in order to get the highest possible accuracy when using a simple CNN's architecture and just a few number of filters.

# MINST Dataset consists of:

  1. 60000 training images whit their labels
  2. 10000 test images with there labels

# Preprocess MNIST database:
  1. The training images have been shifted rundomly to left, right, up and down.
  2. The shifted training images then have be added onto the original training images.
  3. The total number of training images now is 120000.
  
# CNN's Architecture
  1. (Conv -> Act-> Pool) * 2 -> (Flat -> Dense -> Act) -> (Dense -> SoftMax)
  2. Filters in Conv:
  
      2.1. shape = (11, 11)
      2.2. number: 8, 16 respectively
  3. Activation: ReLU
  4. Loss Function: mean_squared_error
  5. Kernels in Dense layer = 16
  6. Epochs = 20

# Results:

**1. Accuracy = 94.69%**

# 2. Filters' Weight:

**2.1. First Conv Layer**
 ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Weights_Conv1.png)
 
 **2.2. Second Conv Layer**
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Weights_Conv2.png)
  
# 3. Activation Maps:

  **3.1. Input Layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/InputLayer.png)
  
  **3.2. First Conv Layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Conv1.png)
  
  **3.3. First MaxPool Layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/MaxPool_1.png)
  
  **3.4. Second Conv Layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Conv_2.png)
  
  **3.5. Second MaxPool layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/MaxPool_2.png)
  
  **3.6. Flatten layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Flatten.png)
  
  **3.7. Output Layer**
  
  ![alt text](https://raw.githubusercontent.com/masjadaan/Convolutional_Neural_Network_CNN/master/CNN_On_MINST/Dense.png)
  
