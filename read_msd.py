import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

''' ---------------------------------- Version 0.1 --------------------------'''

style.use('dark_background')

def read_file(path):
	lines_read = 0
	data = []
	with open(path , 'r' , encoding = 'utf-8') as f:
		for line in f:
			lines_read += 1
			if lines_read > 10: 
				data.append(line.split(','))
	return data

def keep_data(data , features_keep):
	all_features = ['genre','track-id','artist_name','title','loudness','tempo','time_signature','key','mode','duration','avg_timbre1','avg_timbre2','avg_timbre3','avg_timbre4','avg_timbre5','avg_timbre6','avg_timbre7','avg_timbre8','avg_timbre9','avg_timbre10','avg_timbre11','avg_timbre12','var_timbre1','var_timbre2','var_timbre3','var_timbre4','var_timbre5','var_timbre6','var_timbre7','var_timbre8','var_timbre9','var_timbre10','var_timbre11','var_timbre12']
	keep_names = (list(set(all_features).intersection(features_keep)))
	keep_indeces = sorted([all_features.index(i) for i in keep_names])
	indices = list(range(len(all_features)))
	for i in sorted(keep_indeces , reverse = True): 
		del indices[i]
	for x in range(0,len(data)):
		for i in sorted(indices , reverse = True):
			del data[x][i]
	return data


def create_labels(data , one_hot_string = True):
	genres = ['jazz and blues' , 'metal' , 'folk']
	labels = []
	new_data = []
	for i in range(0 , len(data)):
		if (data[i][0] in genres):
			labels.append(genres.index(data[i][0]))
			new_data.append([data[i][x] for x in [1,2]])
	labels = np.array(labels)
	labels_onehot = (np.arange(len(genres)) == labels[: , None]).astype(int)
	#new_labels = []
	#for i in range(0 , len(labels_onehot)):
		#new_labels.append(''.join(str(x) for x in labels_onehot[i]))
	return new_data , labels , labels_onehot

def plot_data(data , labels):
	c = list(zip(data , labels))
	random.shuffle(c)
	data , labels = zip(*c)
	data , labels = data[:int(1/100*len(data))] , labels[:int(1/100*len(labels))]  
	new_labels = []
	colors = ['red' , 'green' , 'blue']
	new_labels = [colors[i] for i in labels]
	x = [data[i][0] for i in range(0,len(data))]
	y = [data[i][1] for i in range(0,len(data))]
	plt.scatter(x , y , c = new_labels)
	plt.show()

def preprocessing(data , labels_onehot , test_percentage = 0.2): 
	all_data = list(zip(data , labels_onehot))
	random.shuffle(all_data)
	data , labels_onehot = zip(*all_data)
	return data[:int(0.2*len(data))] , labels_onehot[:int(0.2*len(labels_onehot))] , data[-int(0.2*len(data)):] , labels_onehot[-int(0.2*len(labels_onehot)):]



def main(): 
	filename = 'C:/Users/nick/Desktop/Dissertation/msd_genre_dataset.txt'
	data = read_file(filename)
	data = keep_data(data , ['genre' , 'tempo' , 'loudness'])
	data , labels , labels_onehot = create_labels(data)
	return preprocessing(data , labels_onehot)

	



if __name__ == '__main__':
	main()

