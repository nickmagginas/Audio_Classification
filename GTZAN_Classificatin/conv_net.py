import filters
import tensorflow
import argparse
import sys


def conv2d(x , W): 
	return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME')

def maxpool2d(x): 
	return tf.nn.maxpool(x , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME')

def network_model(x):
	weights = {'W_conv1':tf.Variable(tf.random_normal([18,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([18,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 480, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output



def main():
	paramemeters = parse_arguments()
	train_features , train_labels , test_features , test_labels = filters.main()
	check(paramemeters)
	build_network_model(paramemeters)
	train_conv_net(train_features , train_labels) 

if __name__ == '__main__':
	main()

