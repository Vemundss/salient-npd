import numpy as np


def histogram_equalization(x,num_colors=256):
	"""
	Do histogram equalization

	Theory source:
	https://www.uio.no/studier/emner/matnat/ifi/INF2310/v20/undervisningsmateriale/forelesning/inf2310-2020-05-histogramoperasjoner.pdf
	"""
	tmp = (x - np.min(x))
	tmp /= np.max(tmp) # x \in [0,1]
	tmp = np.around(tmp * (num_colors-1)) # x \in [0,255] \subseq \mathbb{N}

	n = len(tmp)
	px = np.zeros(num_colors)
	for i in range(num_colors):
		px[i] = np.sum(tmp == i) / n

	cdfx = np.cumsum(px)
	return cdfx[tmp.astype(int)] # [0,1]


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	a = np.abs(np.random.normal(loc=0,scale=1,size=100))
	a = a/np.max(a)

	plt.hist(a,cumulative=True,density=True)
	plt.hist(histogram_equalization(a),cumulative=True,density=True)
	plt.legend(['a','hist_eq(a)'])
	plt.show()