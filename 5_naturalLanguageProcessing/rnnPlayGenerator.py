# Use RNN to create character predictive model that will take as input a variable length sequence and predict the next character
# Can use model many times in row with output from last prediction as input for next call

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

# Dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') # can write own play/poem but we'll use exerpt from Shakespeare play for simplicity
# Alternative way: loading own text file
# from google.colab import files
# path_to_file = list(files.upload().keys())[0] # upload file button, make sure file is .txt

# Read file contents, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])

# Encoding the text; use unique integer to encode each character
vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)} 
idx2char = np.array(vocab) # reverse mapping of index to char

def text_to_int(text): # takes text and converts to int representation
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text) # stores string converted to int into var

# lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

# Decoding: converts int to text
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

# Training examples; takes in input sequence and outputs the sequence shifted over by one(predicts right-most char)
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1) # number of sequences per epoch

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Turns stream of character into batches of desired lengths
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Split sequences into input and output
def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

# Print out 2 examples
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))

# Make the training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Building the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]), # use embedding layer, None means we only know the number of entries in each batch but don't know the length of each entry
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True, # return intermediate steps
                        stateful=True, 
                        recurrent_initializer='glorot_uniform'), # LSTM(long short-term memory) layer, recurent_initializer is what values will start at
    tf.keras.layers.Dense(vocab_size) # dense layer that contains all nodes for each unique char in our training data, gives probability distribution over all nodes
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# Creating loss function
# display sample input/output from untrained model
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape
# we can see that the prediction is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)
# lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step
# and finally well look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probabillity of each character occuring next
# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)
# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)
predicted_chars  # and this is what the model predicted for training sequence 1
# loss function that compares actual output to expected, gives us numeric value representing how close they were
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

