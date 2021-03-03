import numpy as np
import pandas as pd
import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp


class SimpleModel():
   
   def __init__(self, nFeatures, inseed, learn_rate):
       """ Initialize placeholder for a simple ML model
      
       Parameters
       ----------
       nFeatures : int
          Number of input features
       inseed : float
          Seed for setting random initial state of model parameters
       learn_rate : float
          Learning rate used in training the model.

      
      
       """
       self.numfeatures = nFeatures

       self.seed = inseed
      
       self.optimizer=tf.train.AdamOptimizer(learn_rate)
      
      

   def keras_model(self, inputs):
       """ Returns neural network model defined in keras 

       Parameters
       ----------
       inputs : tf.data.Iterator
          Tensor of input data

       Returns
       -------
       tf.keras.Model
            Tensorflow graph for neural network model.

       See Also
       --------
       handwritten_model

       """
       network = tf.keras.Sequential([
         tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(self.numfeatures,)),
         tf.keras.layers.Dense(64, activation=tf.nn.relu),
         tf.keras.layers.Dense(2)
       ])
      
       return network(inputs)
      
      
      
   def handwritten_model(self, inputs):
      """ Returns tensorflow neural network model defined by hand. 
      Parameters
      ----------
      inputs : tf.data.Iterator
         Tensor of input data

      Returns
      -------
      output_layer : tf.Tensor 
            Tensorflow graph for handwritten neural network model.

      See Also
      --------
      keras_model
      load_handwritten_model

      """
      activationf = tf.nn.relu
   
      initializer = tf.initializers.truncated_normal(0, 0.1 ,self.seed)
   
      num_hid1 = self.numfeatures*10
      #num_hid1 = 36*4
      #num_hid2 = num_hid1/2.
      num_hid2 = 15
      num_out = 2
   
      w1 = tf.Variable(initializer([self.numfeatures, num_hid1]), dtype=tf.float32, name='w1')
      w2 = tf.Variable(initializer([num_hid1, num_hid2]), dtype=tf.float32, name='w2')
      w3 = tf.Variable(initializer([num_hid2, num_out]), dtype=tf.float32, name='w3')
    
      b1=tf.Variable(tf.zeros(num_hid1),  name='b1')
      b2=tf.Variable(tf.zeros(num_hid2), name='b2')
      b3=tf.Variable(tf.zeros(num_out), name='b3')
    
      hid_layer1 = activationf(tf.matmul(inputs,w1)+b1)
      hid_layer2 = activationf(tf.matmul(hid_layer1, w2)+b2)
      output_layer = tf.matmul(hid_layer2,w3)+b3
      
      return output_layer

   def load_handwritten_model(self, handle, eval_Iterator, inputs):
      """ Loads and evaluates saved handwritten model.
      Parameters
      ----------
      handle : tf.placeholder(tf.string, shape=[])
      eval_Iterator : tf.data.Iterator
         Object defining the state of iterating over a dataset
      inputs : tf.Tensor
         Tensor of the input data.

      Returns
      -------
      pred  
         List of model predictions

      """
      init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer() )
      with tf.Session() as sess:
         sess.run(init)
         eval_handle = sess.run(eval_Iterator.string_handle())
      
         saver = tf.train.import_meta_graph('./model-10000.meta')
         saver.restore(sess, tf.train.latest_checkpoint('./'))
      
      
         graph = tf.get_default_graph()
         w1 = graph.get_tensor_by_name("w1:0")
         w2 = graph.get_tensor_by_name("w2:0")
         w3 = graph.get_tensor_by_name("w3:0")
      
         b1 = graph.get_tensor_by_name("b1:0")
         b2 = graph.get_tensor_by_name("b2:0")
         b3 = graph.get_tensor_by_name("b3:0")
      
         activationf = tf.nn.relu
      
         hid_layer1 = activationf(tf.matmul(inputs,w1)+b1)
         hid_layer2 = activationf(tf.matmul(hid_layer1, w2)+b2)
         output_layer = tf.matmul(hid_layer2,w3)+b3
      
         predictions = tf.argmax(output_layer, axis=1)
      
         pred = sess.run(predictions, feed_dict={handle: eval_handle})
      
      
      return pred
   
   
