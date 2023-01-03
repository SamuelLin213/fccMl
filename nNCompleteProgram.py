# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training sets

# here, we create an array of label names to match label to clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 

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
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]
  print("Expected: " + class_names[correct_label])
  print("Actual: " + predicted_class)

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Expected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
