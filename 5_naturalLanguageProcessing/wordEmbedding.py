# word encoding keeps order of word intact as well as encoding similar words w/ very similar labels
# attempts to encode not just frequency/order of words, but also the meaning of the word in a sentence

# import libraries
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE) # loads in IMDB review dataset from keras

# Lets look at one review
train_data[1]

# Notice how each review has diff length; we have to make each review length the same:
    # if review len > 250: trim off extra words
    # if review len < 250: add necessary amount of 0's to make it equal to 250
train_data = pad_sequences(train_data, MAXLEN)
test_data = pad_sequences(test_data, MAXLEN)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), # word embedding layer, outputs vector of 32 dimensions
    tf.keras.layers.LSTM(32), # LSTM layer
    tf.keras.layers.Dense(1, activation="sigmoid") # dense node, sigmoid squeezes output between 0 and 1
])
model.summary()

# training model
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# evaluate model; look at results
results = model.evaluate(test_data, test_labels)
print(results)

# making predictions
word_index = imdb.get_word_index()

def encode_text(text): # convert review into encoded form so network can understand
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers): # converts integers back to review string
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))

# now time to make a prediction

def predict(text):
  encoded_text = encode_text(text) # encode passed in text
  pred = np.zeros((1,250)) # create blank numpy array 
  pred[0] = encoded_text # stores encoded string into array
  result = model.predict(pred) 
  print(result[0])
  if result[0] > 0.5:
    print("Positive")
  else:
    print("Negative")

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)
