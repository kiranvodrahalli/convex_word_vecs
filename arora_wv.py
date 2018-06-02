import numpy as np 
from scipy.io import loadmat


fname = '/Users/kiranv/college/3junior-year/spring2015/cos511/project/arora_words_vecs.mat'

matlab_dict = loadmat(fname)

W = matlab_dict['W']
words = matlab_dict['words']

# goes from word to index
arora_word_dict = dict()
arora_index_dict = dict()
for i in range(len(words)):
	word = words[i][0][0]
	if word not in arora_word_dict:
		arora_word_dict[word] =  i
	if i not in arora_index_dict:
		arora_index_dict[i] = word
