import tensorflow as tf
import read_msd
import numpy as np

x = tf.placeholder(dtype = 'float32' , shape = [None , 30])
y = tf.placeholder(dtype = 'float32' , shape = [None , 7])
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = 7
batch_size = 50



def neural_network(x):
	hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([30 , n_nodes_hl1])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_layer_2 = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
	output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2 , n_classes])) , 'biases' : tf.Variable(tf.random_normal([n_classes]))}
	l1 = tf.add(tf.matmul(x , hidden_layer_1['weights']) , hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1 , hidden_layer_2['weights']) , hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)
	output = tf.add(tf.matmul(l2 , output_layer['weights']) , output_layer['biases'])
	return output

def train_network(train_x , train_y , test_x , test_y , lenght): 
	prediction = neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	epoch_number = 10
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())

		for epoch in range(0 , epoch_number): 
			epoch_loss = 0
			batches_sent = 0
			for _ in range(0 , int(lenght/batch_size)):
				epoch_x , epoch_y = get_next_batch(train_x , train_y , batch_size , batches_sent)
				_ , c = sess.run([optimizer , cost] , feed_dict = {x : epoch_x , y : epoch_y})
				epoch_loss += c
				batches_sent += 1
			epoch_loss = c
			print('Epoch' , epoch , 'completed out of' , epoch_number , 'loss :' , epoch_loss)

		correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(y , 1))
		accuracy = tf.reduce_mean(tf.cast(correct , 'float'))
		print('Accuracy' , accuracy.eval({x : test_x , y : test_y}))


def get_next_batch(X , y , batch_size , batches_read):
	return X[batches_read*batch_size : (batches_read + 1) * batch_size] , np.array(y[batches_read*batch_size : (batches_read + 1) * batch_size])

def main():
	X , y , test_x , test_y , lenght = read_msd.main()
	X , y = np.array(X).astype(np.float32) , np.array(y).astype(np.float32)
	train_network(X , y , test_x , test_y , lenght)



if __name__ == '__main__': 
	main()



