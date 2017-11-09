import filters 
import tensorflow as tf
import numpy as np

global X
global Y

X , Y , test_X , test_Y , num_examples = filters.main()
X = np.array(X).astype(np.float32)

n_classes = 10

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

y = tf.placeholder('float' , [None , 10])
x = tf.placeholder('float' , [None , 13440])

batch_size = 64

def get_next_batch(i , batch_size): 
  return X[int(i*batch_size) , int((i+1)*batch_size)] , Y[int(i*batch_size) , int((i+1)*batch_size)] 


def conv2d(x , W): 
  return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME')

def maxpool2d(x): 
  return tf.nn.max_pool(x , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME')

def neural_network_model(x): 
  print(x.shape)
  weights = {'W_Conv1' : tf.Variable(tf.random_normal([5,5,1,534])),
              'W_Conv2' : tf.Variable(tf.random_normal([5,5,534,1068])),
              'W_FC' : tf.Variable(tf.random_normal([120*7*1068, 1024])),
              'Out'  : tf.Variable(tf.random_normal([1024,n_classes]))
   }

  biases = {'B_Conv1' : tf.Variable(tf.random_normal([534])),
              'B_Conv2' : tf.Variable(tf.random_normal([1068])),
              'B_FC' : tf.Variable(tf.random_normal([1024])),
              'Out'  : tf.Variable(tf.random_normal([n_classes]))
   }

  x = tf.reshape(x , shape = [-1 , 480 , 28 , 1])

  conv1 = tf.nn.relu(conv2d(x, weights['W_Conv1']) + biases['B_Conv1'])
  print(conv1.shape)
  conv1 = maxpool2d(conv1)
  print(conv1.shape)
  
  conv2 = tf.nn.relu(conv2d(conv1, weights['W_Conv2']) + biases['B_Conv2'])
  print(conv2.shape)
  conv2 = maxpool2d(conv2)
  print(conv2.shape)

  fc = tf.reshape(conv2,[-1, 120*7*1068])
  fc = tf.nn.relu(tf.matmul(fc, weights['W_FC'])+biases['B_FC'])
  fc = tf.nn.dropout(fc, keep_rate)

  output = tf.matmul(fc, weights['Out'])+biases['Out']

  return output

def train_network(x , test_x , test_y , num_examples , batch_size): 
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)

  hm_epochs = 10

  with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    for epoch in range(0 , hm_epochs): 
      epoch_loss = 0 
      received = 0
      for _ in range(num_examples/batch_size): 
        epoch_x , epoch_y = get_next_batch(received , batch_size)
        _ , c = sess.run([optimizer , cost] , feed_dict = {x : epoch_x , y : epoch_y})
        epoch_loss += c
        received += 1 

      print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

  correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(y , 1))
  accuracy = tf.reduce_mean(tf.cast(correct , 'float'))
  print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


train_network(X , test_X , test_Y , num_examples , batch_size)