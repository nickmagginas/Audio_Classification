import numpy as np 
import soundfile as sf
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import simps
import random
import create_image
 

style.use('dark_background')

def audioread(filename):
	data , fs = sf.read(filename)
	data = data[:660000]
	return data , fs

def find_nearest(x , value): 
	for i in range(0 , len(x)):
		if x[i] - value > 0:
			if abs(x[i] - value) < abs(x[i - 1] - value):
				return x[i]
			return x[i-1]

def get_loudness(data): 
	return np.average(data)


def furrier_transform(data , fs):
	L = len(data)
	y = fft(data)
	P = abs(y/L)
	P = P[:int(L/2)]
	P = [2*i for i in P]
	f = [fs/L*i for i in list(range(0,int(L/2)))]
	return {'Amplitude' : P , 'Frequency' : f}

def filter(data , cutoff = None , cut_low = None):
	new_data = {'Amplitude' : [] , 'Frequency' : []}
	for i in range(0 , len(data['Frequency'])):
		if data['Frequency'][i] < cutoff and data['Frequency'][i] > cut_low :
			new_data['Amplitude'].append(data['Amplitude'][i])
			new_data['Frequency'].append(data['Frequency'][i])

	return new_data

def calculate_area(data , Simpsons = True): 
	if Simpsons:
		return simps(data['Amplitude'] , dx = (data['Frequency'][1] - data['Frequency'][0]))
	return np.trapz(data['Amplitude'] , dx = (data['Frequency'][1] - data['Frequency'][0]))

def discrete_spectral_density(data , spacing = 50 , cutoff = 3920):
	areas = []
	for i in range(0 , cutoff , spacing):
		areas.append(calculate_area(filter(data , i + spacing , i) , Simpsons = False))
	return areas

def plot(y , data , spacing = 50 , cutoff = 3920):
	x = list(range(0 , cutoff , spacing))
	plt.plot(x , y)
	plt.xlabel('Frequency(Hz)')
	plt.ylabel('Spectral Density')
	plt.grid(color = 'r')
	plt.show()

def analyze_sample(filename):
	data , fs = audioread(filename)
	data = furrier_transform(data , fs)
	spectral_info = discrete_spectral_density(data , spacing = 5)
	return spectral_info


def get_data(one_hot = False):
	print('Beggining audio file reading')
	audio_data = []
	labels = []
	genres = ['blues'] # , 'classical', 'country' , 'disco' , 'hiphop' , 'jazz' , 'metal' , 'pop' , 'reggae' , 'rock']
	#main_path = '/home/nick/Desktop/Dissertation/Audio_Classification/GTZAN_Classificatin/Genres'
	main_path = 'C:/Users/nick/Desktop/Dissertation/Audio_Classification/GTZAN_Classificatin/Genres'
	n_samples = 100

	for x in range(0 , len(genres)): 
		genre_path = main_path + '/' + genres[x]
		for i in range(0 , n_samples):
			current_count = '.' + '%05d' % i
			song_path = genre_path + '/' + genres[x] + current_count + '.au'
			audio_data.append(create_image.main(song_path))
			labels.append(x)
			if i%10 == 0:
				print('Percentage Complete :' , int(((x*100)+i)/(len(genres)*100)*100) , '%')
	print('Audio read succesfully')

	labels = np.array(labels)
	labels_onehot = (np.arange(len(genres)) == labels[: , None]).astype(int)

	return audio_data , labels_onehot

def create_dataset(x , y , test_percentage = 0.2): 
	all_data = list(zip(x , y))
	random.shuffle(all_data)
	x , y = zip(*all_data)
	return x[:int(test_percentage*len(x))] , y[:int(test_percentage*len(y))] , x[-int(test_percentage*len(x)):] , y[-int(test_percentage*len(y)):] , len(x)


def main(): 
	x , y = get_data(one_hot = True)
	return create_dataset(x , y)
	

if __name__ == '__main__':
	main()


