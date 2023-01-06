import tensorflow as tf  # now import the tensorflow module

# tensor is generalization of vector/matrices to higher dimensions; each has data type and shape
# data types: float32, int32, string, others
# shape: represents dimension of data

# Creating tensors; note that it's rare to see string tensor
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# rank/degree: number of dimensions involved in tensor; created tensor of rank 0 above(scalar), rank related to deepest level of nesting
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
tf.rank(rank2_tensor) # determines rank of tensor

# shape of tensor: number of elements that exist in each dimension
rank2_tensor.shape

# changing shape of tensor
tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,3]                                               
# The number of elements in the reshaped tensor MUST match the number in the original; number of elements in tensor is product of sizes of all its shapes
