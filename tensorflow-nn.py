import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_clean import PURPOSE

class ModelTraining(object):
  def __init__(self, X, Y, num_of_layer=None):
    self.parameters = {}
    self.shape = []
    self.input_X = X
    self.input_Y = Y
    self.num_of_layer = num_of_layer
    self.num_of_node = []

  def create_placeholders(self, num_features, num_labels):
    data_set = tf.placeholder(tf.float32, shape=(None, num_features), name='X')
    label_set = tf.placeholder(tf.float32, shape=(None, num_labels), name='Y')

    return data_set, label_set

  def _shape_calc(self):
    first = ((35, 30), (30))
    second = ((30, 20), (20))
    third = ((20, 14), (14))

    self.num_of_node.append(first)
    self.num_of_node.append(second)
    self.num_of_node.append(third)

  def initialize_parameters(self):
    ''' initialize '''
    self.parameters['W1'] = tf.Variable(tf.random_normal((35, 14)))
    self.parameters['b1'] = tf.Variable(tf.zeros(14))

    # self.parameters['W2'] = tf.Variable(tf.random_normal((30, 20)))
    # self.parameters['b2'] = tf.Variable(tf.zeros(20))

    # self.parameters['W3'] = tf.Variable(tf.random_normal((20, 14)))
    # self.parameters['b3'] = tf.Variable(tf.zeros(14))

  def forward_propagation(self, init_input):
    ''' forward propagation '''
    num = len(self.parameters)//2 + 1
    for i in range(1, num):
      input_data = init_input if i == 1 else hidden_output
      bais = self.parameters['b{}'.format(i)]
      weight = self.parameters['W{}'.format(i)]

      hidden_output = tf.add(tf.matmul(input_data, weight), bais) 
      output_func = tf.nn.softmax if i == num - 1 else tf.nn.relu
    
      hidden_output = output_func(hidden_output)                                     
      print input_data.shape,hidden_output.shape
    return hidden_output

  def compute_cost(self, hidden_output, label_holder):
    logits = tf.transpose(hidden_output)
    labels = tf.transpose(label_holder)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

  def model(self, learning_rate=0.01,num_epochs=1500, minibatch_size=2, print_cost=True):
    tf.reset_default_graph() 
    print self.input_X.shape                                                               
    num_samples, num_features = self.input_X.shape                          
    num_labels = len(self.input_Y.unique())                           
    costs = []                                      
    data_holder, label_holder = self.create_placeholders(num_features, num_labels)

    self.initialize_parameters()
    hidden_output = self.forward_propagation(data_holder)
    cost = self.compute_cost(hidden_output, label_holder)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      random_data, random_label = tf.train.slice_input_producer([self.input_X, self.input_Y], shuffle=True)
      num_minibatches = int(num_samples / minibatch_size) 
      print_costs = []
      print 'begin - mini-batch, size {}'.format(num_minibatches)   
      for epoch in range(num_epochs):
        epoch_cost = 0.             
        for i in range(num_minibatches):   
          print i     
          minibatch_data, minibatch_label = tf.train.batch([random_data, random_label], batch_size=minibatch_size)
          _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_data.eval(), Y: minibatch_label.eval()})
            
          epoch_cost += minibatch_cost / num_minibatches
        if epoch % 10 == 0:
          print 'Cost after epoch {}: {}'.format(epoch, epoch_cost)
        if epoch % 5 == 0:
          print_costs.append(epoch_cost)

      parameters = sess.run(self.parameters)
      
      print 'Parameters have been trained'
      correct_prediction = tf.equal(tf.argmax(hidden_output), tf.argmax(label_holder))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

      saver = tf.train.Saver()
      saver.save(sess, 'nn_model.ckpt')
      
      return self.parameters


if __name__ == '__main__':
  df = pd.read_csv('processed.csv')
  y = df.purpose.T
  x = df.drop('purpose', axis=1)
  trainer = ModelTraining(x,y)
  trainer.model()
