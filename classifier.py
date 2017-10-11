import tensorflow as tf
import read_msd

x = tf.placeholder('float')
y = tf.placeholder('float')
n_nodes_hl1 = 100
n_classes = 3
batch_size = 50



def neural_network(X): 
	hidden_layer = {'weights' : tf.Variable(tf.random_normal([2 , n_nodes_hl1])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
	output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1 , n_classes])) , 'biases' : tf.Variable(tf.random_normal([n_classes]))}
	l1 = tf.add(tf.matmul(x , hidden_layer['weights']) , hidden_layer['biases'])
	l1 = tf.nn.relu(l1)
	output = tf.add(tf.matmul(l1 , output_layer['weights']) , output_layer['biases'])
	return output

def train_network(X , y , test_x , test_y): 
	prediction = neural_network(X)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	epoch_number = 10
	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())

		for epoch in range(0 , epoch_number): 
			epoch_loss = 0
			batches_sent = 0
			for _ in range(0 , int(len(X)/batch_size)):
				epoch_x , epoch_y = get_next_batch(X , y , batches_size , batches_sent)
				_ , c = sess.run([optimizer , cost] , feed_dict = {x : epoch_x , y : epoch_y})
				epoch_loss += c
				batches_sent += 1

			print('Epoch' , epoch , 'completed out of' , epoch_number , 'loss :' , epoch_loss)

		correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(y , 1))
		accuracy = tf.reduce_mean(tf.cast(correct , 'float'))
		print('Accuracy' , accuracy.eval({x : test_x , y : test_y}))


def get_next_batch(X , y , batch_size , batches_read):
	return X[batches_read*batch_size : (batches_read + 1) * batch_size] , y[batches_read*batch_size : (batches_read + 1) * batch_size] 


def main():
	X , y , test_x , test_y = read_msd.main()
	train_network(X , y , test_x , test_y)



if __name__ == '__main__': 
	main()



