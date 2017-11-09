import filters
import numpy as np 
import soundfile as sf
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib import style
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
	#print('Dimension test passsed')

def plot_data(segments_transformed , stacked = True): 

	if not stacked :
		for i in range(0 , len(segments_transformed)): 
			x = segments_transformed[i]['Frequency']
			y = segments_transformed[i]['Amplitude']
			plt.plot(x , y)
			plt.show()
			return

	dicplot={}
	fig = plt.figure()
	for i in range(1 , 29):

		dicplot[str(i)] = fig.add_subplot(28,1,i)
		dicplot[str(i)].plot(segments_transformed[i-1]['Frequency'] , segments_transformed[i-1]['Amplitude'])

	plt.show()

def create_image(data , filename): 
	data = [data[i]['Amplitude'] for i in range(0 , len(data))]
	test_data = data[0]
	for i in range(1 , len(data)):
		test_data.extend(data[i])
	maximum = np.max(test_data)
	mult = lambda x : (int(x * 255/maximum)) #, int(x * 255/maximum) , int(x * 255/maximum))
	pixel_data = list(map(mult , test_data))
	'''
	img = Image.new('RGB' , (480 , 28))
	img.putdata(pixel_data)
	img.save(filename + '.png')
	'''
	return pixel_data





def main(filename):
	data , fs = audioread(filename)
	segments = create_segments(data , fs)
	segments_transformed = [filters.filter(filters.furrier_transform(segments[i],fs) , 481 , 0) for i in range(0 , 28)]
	frequency_concistency_test(segments_transformed)
	return create_image(segments_transformed , filename)
	



if __name__ == '__main__': 
	main()


