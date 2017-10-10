import numpy as np 
import soundfile as sf
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib import style

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
			print(data['Frequency'][i])
			new_data['Amplitude'].append(data['Amplitude'][i])
			new_data['Frequency'].append(data['Frequency'][i])

	return new_data




def main(): 
	filename = 'C:\\Users\\nick\\Desktop\\Dissertation\\Genres\\classical\\classical.00019.au'
	data , fs = audioread(filename)
	data = furrier_transform(data , fs)
	data = filter(data , 100 , 50)
	plt.plot(data['Frequency'] , data['Amplitude'])
	plt.show()






if __name__ == '__main__':
	main()
