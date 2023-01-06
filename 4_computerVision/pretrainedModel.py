# In addition to using data augmentation to expand generalization of model, a pretrained model helps fine tunes classication

#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation, each image has 3 color channels
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels, takes integer input and returns label string

# display 2 images from the dataset
for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  plt.show()

# note how each of the images has different dimensions; we need to scale them to the same size
IMG_SIZE = 160 # All images will be resized to 160x160; it's better to go smaller than larger

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

# apply the format_example() func to all the images through map()
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# let's look at resized images
for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  plt.show()

# finally we can shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Print out the sizes to compare before and after
for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)

# Specify the convolutional base for the model; in this case, we'll use the MobileNet V2 developed at Google, trained on 1.4 million images and 1000 diff classes
# When we use this model, we need to specify that we only want the convolutional base and not the classification/top layer
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, # don't want the top layer
                                               weights='imagenet')
                                        
for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape) # outputs (32, 5, 5, 1280), which is feature extraction from original (1, 160, 160, 3)

# Important:freeze the base, to prevent the weights of the layer from being retrained again(don't want to retrain all that data)
base_model.trainable = False

# add our own classifier; use a global average pooling layer that'll average entire 5x5 area of each 2D feature map and return single 1280 vector per filter
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1) # add the prediction layer which will be a single dense neuron(can do this since we only have 2 classes)
# Combine the these layers together in a model
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
]) # only need to modify weights of global avg and prediction layers

# Train and compile model; use a low learning rate to ensure model doesn't make major changes to it
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # use BinaryCrossentropy as there's only 2 classes
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')
