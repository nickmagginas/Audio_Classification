import numpy as np 
import soundfile as sf
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import simps

style.use('dark_background')

def audioread(filename):
	data , fs = sf.read(filename)
	data = data[:66000]
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

def discrete_spectral_density(data , spacing = 50):
	areas = []
	for i in range(0 , int(np.max(data['Frequency'])) , spacing):
		areas.append(calculate_area(filter(data , i + spacing , i) , Simpsons = False))
	return areas

def plot(y , data , spacing = 50):
	x = list(range(0 , int(np.max(data['Frequency'])) , spacing))
	plt.plot(x , y)
	plt.xlabel('Frequency(Hz)')
	plt.ylabel('Spectral Density')
	plt.grid(color = 'r')
	plt.show()




def main(): 
	filename = 'C:\\Users\\nick\\Desktop\\Dissertation\\Genres\\jazz\\jazz.00024.au'
	data , fs = audioread(filename)
	data = furrier_transform(data , fs)
	spectral_info = discrete_spectral_density(data , spacing = 1)
	plot(spectral_info , data , spacing = 1)
	







if __name__ == '__main__':
	main()
