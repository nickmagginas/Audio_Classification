import filters
import numpy as np 
import soundfile as sf
from scipy.fftpack import fft
from filters import furrier_transform
import matplotlib.pyplot as plt
from matplotlib import style
from filters import filter
from PIL import Image

style.use('dark_background')

def audioread(filename): 
	data , fs = sf.read(filename)
	data = data[:660000]
	return data , fs

def create_segments(data , fs , segment_duration = 1): 
	segments = [data[segment_duration*i*fs : segment_duration*(i + 1)*fs] for i in range(0 , 28)]
	return segments

def frequency_concistency_test(segments_transformed): 
	for i in range(0 , (len(segments_transformed)-1)): 
		if segments_transformed[i]['Frequency'] != segments_transformed[i+1]['Frequency']: 
			print('Dimension test failed')
			return
	print('Dimension test passsed')

def plot_data(segments_transformed): 
	for i in range(0 , len(segments_transformed)): 
		x = segments_transformed[i]['Frequency']
		y = segments_transformed[i]['Amplitude']
		plt.plot(x , y)

	plt.show()



def main():
	filename = 'C:\\Users\\nick\\Desktop\\Dissertation\\Genres\\jazz\\jazz.00012.au'
	data , fs = audioread(filename)
	segments = create_segments(data , fs)
	segments_transformed = [filter(furrier_transform(segments[i],fs) , 2000 , 0) for i in range(0 , 28)]
	frequency_concistency_test(segments_transformed)

	plot_data(segments_transformed)



if __name__ == '__main__': 
	main()