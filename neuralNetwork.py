# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training sets

# print(train_images.shape) # prints out shape of data

# print(train_images[0,23,23]) # looks at one pixel of image
    # one pixel has value between 0 and 255, where 0 is black and 255 is white

# print(train_labels[:10]) # first 10 training labels, each has value between 0 and 9, each int represents an article of clothing

# here, we create an array of label names to match label to clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print out image
# plt.figure()    
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# before creating our model, we preprocess the data: apply some prior transformations to data before feeding to model
# in this case, we'll scale all greyscale pixel values(0-255) to be between 0 and 1; divide each value in set by 255.0; smaller values will make it easier to process our values
train_images = train_images / 255.0
test_images = test_images / 255.0
# make sure you preprocess both test and training data

# Creating our model
model = keras.Sequential([ # info passing from left to right
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1), flattens matrix into 784 length list
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2), dense layer(all nodes in prev layer are connected to this layer), 128 neurons, activation function is rectified linear unit
    keras.layers.Dense(10, activation='softmax') # output layer (3), dense layer, 10 output neurons, matches number of outputs, softmax ensure all neurons add up to one and are between 0 and 1
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=3)  # fit model to training data, accuracy applies from the training data

# Evaluate precision of data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc) # note how the accuracy of the test data is lower than that of the training data; this is called overfitting

# Making predictions
predictions = model.predict(test_images) # make prediction by passing in array of data to .predict(), returns array of predictions
# print(predictions[0]) # prints list containing predictions for first image
# print(np.argmax(predictions[0])) # returns highest value in predictions of first image
print(class_names[np.argmax(predictions[1])]) # passes in index of highest prediction into class_names array
plt.figure()    
plt.imshow(test_images[1]) # checking the nth test image
plt.colorbar()
plt.grid(False)
plt.show()