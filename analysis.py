import numpy as np
from numpy.linalg import norm
from adagrad import adagrad
from adagrad import n
from adagrad import k
import pickle
import heapq as hq
from sklearn.manifold import TSNE
from saving import save
import matplotlib.pyplot as plt


# get data
word_dict = pickle.load(open("word_dict3.p", "rb"))
index_dict = pickle.load(open("index_dict3.p", "rb"))
input_list = pickle.load(open("input_list3.p", "rb"))

print 'first load'


# function for evaluating the answer to the analogy question 
# maximize ||Ax_j||_2^2 (argmax_j ||-v_a + v_b + v_c + v_j||_2^2) should yield j = y, since
# we penalize large values for all j \neq y. 
# A_mat is the n x k matrix with each row vector a word vector. #needs to be a map from words to vectors.
# this map is simply word_dict, since it maps a word to an index (this index is the one used to build the matrix)
def query(A_mat, w1, w2, w3):
	# indices of columns
	j1 = word_dict[w1]
	j2 = word_dict[w2]
	j3 = word_dict[w3]
	v1 = A_mat[j1] 
	v2 = A_mat[j2]
	v3 = A_mat[j3]
	argmax = 0
	max_val = 0
	for i in range(0, n):
		# need to restrict so max val in input is 1 or -1
		if (i != j1) and (i != j2) and (i != j3):
			v4 = A_mat[i]
			objective = norm(-1*v1 + v2 + v3 + v4)**2
			if objective > max_val:
				max_val = objective
				argmax = i
	# return the best fit word and its objective value.
	return index_dict[argmax], max_val

# query where indices are presented instead of words
def query_top1(A_mat, j1, j2, j3):
	v1 = A_mat[j1] 
	v2 = A_mat[j2]
	v3 = A_mat[j3]
	argmax = 0
	max_val = 0
	for i in range(0, n):
		# need to restrict so max val in input is 1 or -1
		if (i != j1) and (i != j2) and (i != j3):
			v4 = A_mat[i]
			objective = norm(-1*v1 + v2 + v3 + v4)**2
			if objective > max_val:
				max_val = objective
				argmax = i
	# return the best fit word and its objective value.
	return argmax, max_val

def is_right(A_mat, j1, j2, j3, ans):
	argmax, max_val = query_top1(A_mat, j1, j2, j3)
	if argmax == ans:
		return 1
	else:
		return 0

# query where indices are presented instead of words
# returns topk results in a set
def query_topk(A_mat, j1, j2, j3, topk):
	v1 = A_mat[j1] 
	v2 = A_mat[j2]
	v3 = A_mat[j3]
	topk_argmaxes = []
	smallest_maxval = float('inf')
	for i in range(0, n):
		# need to restrict so max val in input is 1 or -1
		if (i != j1) and (i != j2) and (i != j3):
			v4 = A_mat[i]
			objective = norm(-1*v1 + v2 + v3 + v4)**2
			if len(topk_argmaxes) < topk:
				hq.heappush(topk_argmaxes, (objective, i))
				if objective < smallest_maxval:
					smallest_maxval = objective
			elif objective > smallest_maxval:
				val, index = hq.heappop(topk_argmaxes)
				smallest_maxval = topk_argmaxes[0][0]
				if objective < smallest_maxval:
					smallest_maxval = objective
				hq.heappush(topk_argmaxes, (objective, i))
	# return the best fit word and its objective value.
	return set(map(lambda (a, b): b, topk_argmaxes))

# returns 1 if answer is in the top k answers
# k > 1
def topk_performance(A_mat, j1, j2, j3, topk, ans):
	topk_argmaxes = query_topk(A_mat, j1, j2, j3, topk)
	if ans in topk_argmaxes:
		return 1
	else:
		return 0

# split into training testing
train = input_list[0:1000]
test = input_list[1000:2000]

'''
# use AdaGrad to learn word vectors 
A = adagrad(train)
# save word vectors
pickle.dump(A, open('A.p', 'wb'))
'''

'''
# load the trained vectors
A = pickle.load(open('A1000.p', 'rb'))

# use query function to test train/test accuracies 
# training accuracy
train_correct1 = sum(is_right(A, t[0], t[1], t[2], t[3]) for t in train) / (0. + len(train))
train_correct5 = sum(topk_performance(A, t[0], t[1], t[2], 5, t[3]) for t in train) / (0. + len(train))
train_correct10 = sum(topk_performance(A, t[0], t[1], t[2], 10, t[3]) for t in train) / (0. + len(train))
train_correct20 = sum(topk_performance(A, t[0], t[1], t[2], 20, t[3]) for t in train) / (0. + len(train))

print "-------------------------------------------------"
print "Training Accuracy: "
print "Top 1 Accuracy Score: " + str(train_correct1)
print "Top 5 Accuracy Score: " + str(train_correct5)
print "Top 10 Accuracy Score: " + str(train_correct10)
print "Top 20 Accuracy Score: " + str(train_correct20)

# testing accuracy
test_correct1 = sum(is_right(A, t[0], t[1], t[2], t[3]) for t in test) / (0. + len(test))
test_correct5 = sum(topk_performance(A, t[0], t[1], t[2], 5, t[3]) for t in test) / (0. + len(test))
test_correct10 = sum(topk_performance(A, t[0], t[1], t[2], 10, t[3]) for t in test) / (0. + len(test))
test_correct20 = sum(topk_performance(A, t[0], t[1], t[2], 20, t[3]) for t in test) / (0. + len(test))

print "-------------------------------------------------"
print "Testing Accuracy: "
print "Top 1 Accuracy Score: " + str(test_correct1)
print "Top 5 Accuracy Score: " + str(test_correct5)
print "Top 10 Accuracy Score: " + str(test_correct10)
print "Top 20 Accuracy Score: " + str(test_correct20)

print "-------------------------------------------------"
'''

'''
train50 = input_list[0:50]
test50 = input_list[50:100]

# use query function to test train/test accuracies 
# training accuracy
train50_correct1 = sum(is_right(A, t[0], t[1], t[2], t[3]) for t in train50) / (0. + len(train50))
train50_correct5 = sum(topk_performance(A, t[0], t[1], t[2], 5, t[3]) for t in train50) / (0. + len(train50))
train50_correct10 = sum(topk_performance(A, t[0], t[1], t[2], 10, t[3]) for t in train50) / (0. + len(train50))
train50_correct20 = sum(topk_performance(A, t[0], t[1], t[2], 20, t[3]) for t in train50) / (0. + len(train50))

print "-------------------------------------------------"
print "Training Accuracy: "
print "Top 1 Accuracy Score: " + str(train50_correct1)
print "Top 5 Accuracy Score: " + str(train50_correct5)
print "Top 10 Accuracy Score: " + str(train50_correct10)
print "Top 20 Accuracy Score: " + str(train50_correct20)

# testing accuracy
test50_correct1 = sum(is_right(A, t[0], t[1], t[2], t[3]) for t in test50) / (0. + len(test50))
test50_correct5 = sum(topk_performance(A, t[0], t[1], t[2], 5, t[3]) for t in test50) / (0. + len(test50))
test50_correct10 = sum(topk_performance(A, t[0], t[1], t[2], 10, t[3]) for t in test50) / (0. + len(test50))
test50_correct20 = sum(topk_performance(A, t[0], t[1], t[2], 20, t[3]) for t in test50) / (0. + len(test50))

print "-------------------------------------------------"
print "Testing Accuracy: "
print "Top 1 Accuracy Score: " + str(test50_correct1)
print "Top 5 Accuracy Score: " + str(test50_correct5)
print "Top 10 Accuracy Score: " + str(test50_correct10)
print "Top 20 Accuracy Score: " + str(test50_correct20)

print "-------------------------------------------------"
'''


# plot with t-SNE
# mat = A
# only plot with n = 4300
def plot_tsne(mat):
	model = TSNE(n_components=2, random_state=0)
	# n x 2
	t_mat = model.fit_transform(mat)
	x = [t_mat[i][0] for i in range(0, n)]
	y = [t_mat[i][1] for i in range(0, n)]
	plt.scatter(x, y)
	save("/Users/kiranv/college/3junior-year/spring2015/cos511/project/paper/figures/tsne_plt", ext='jpg')
	plt.show()



# compare to arora word vectors 
arora_loc = "/Users/kiranv/college/3junior-year/spring2015/cos511/large_project_files/"
arora_index_dict = pickle.load(open(arora_loc + "arora_index_dict.p", 'rb'))
arora_word_dict = pickle.load(open(arora_loc + "arora_word_dict.p", 'rb'))
arora_wvs = pickle.load(open(arora_loc + "arora_wvs.p", 'rb'))

'''
# reduce arora word vectors to 4300: the words used in this project
A_arora = np.zeros((n, 300))
for i in index_dict.keys():
	w = index_dict[i]
	if w in arora_word_dict:
		i_arora = arora_word_dict[w]
		A_arora[i] = arora_wvs[i_arora]

pickle.dump(A_arora, open("arora_wvs4300.p", 'wb'))
'''


A_arora = pickle.load(open("arora_wvs4300.p", 'rb'))

A1000 = pickle.load(open("A1000.p", 'rb'))


print 'loaded 1'

# use query function to test train/test accuracies 
# training accuracy
train1000_correct1 = sum(is_right(A1000, t[0], t[1], t[2], t[3]) for t in train) / (0. + len(train))
train1000_correct5 = sum(topk_performance(A1000, t[0], t[1], t[2], 5, t[3]) for t in train) / (0. + len(train))
train1000_correct10 = sum(topk_performance(A1000, t[0], t[1], t[2], 10, t[3]) for t in train) / (0. + len(train))
train1000_correct20 = sum(topk_performance(A1000, t[0], t[1], t[2], 20, t[3]) for t in train) / (0. + len(train))

print "-------------------------------------------------"
print "Training Accuracy for 1000 Iterations: "
print "Top 1 Accuracy Score: " + str(train1000_correct1)
print "Top 5 Accuracy Score: " + str(train1000_correct5)
print "Top 10 Accuracy Score: " + str(train1000_correct10)
print "Top 20 Accuracy Score: " + str(train1000_correct20)

# testing accuracy
test1000_correct1 = sum(is_right(A1000, t[0], t[1], t[2], t[3]) for t in test) / (0. + len(test))
test1000_correct5 = sum(topk_performance(A1000, t[0], t[1], t[2], 5, t[3]) for t in test) / (0. + len(test))
test1000_correct10 = sum(topk_performance(A1000, t[0], t[1], t[2], 10, t[3]) for t in test) / (0. + len(test))
test1000_correct20 = sum(topk_performance(A1000, t[0], t[1], t[2], 20, t[3]) for t in test) / (0. + len(test))

print "-------------------------------------------------"
print "Testing Accuracy for 1000 Iterations: "
print "Top 1 Accuracy Score: " + str(test1000_correct1)
print "Top 5 Accuracy Score: " + str(test1000_correct5)
print "Top 10 Accuracy Score: " + str(test1000_correct10)
print "Top 20 Accuracy Score: " + str(test1000_correct20)

print "-------------------------------------------------\n\n\n\n\n\n\n\n"

# takes an index from my word vectors,
# finds the associated word, 
# finds the index of the word in arora's dictionary.
def map_to_arora_index(this_index):
	word = index_dict[this_index]
	if word not in arora_word_dict:
		print word + " not in Arora's word vector set\n"
		return -1
	arora_index = arora_word_dict[word]
	return arora_index

# we want to compare arora and our own input
'''
arora_input_list = []
# kiran testing input
valid_testing_input_list = []
for i in range(0, 10000):
	t = input_list[i]
	t1, t2, t3, t4 = map_to_arora_index(t[0]), map_to_arora_index(t[1]), map_to_arora_index(t[2]), map_to_arora_index(t[3])
	if not (t1 == -1 or t2 == -1 or t3 == -1 or t4 == -1):
		arora_input_list.append((t1, t2, t3, t4))
		valid_testing_input_list.append(t)
'''
# ~ 3979 (w1, w2, w3, w4) that will work in arora's dictionary
arora_test = pickle.load(open("arora_testing_inputs.p", 'rb'))
# corresponding 3979 that my word vectors can compare to
kiran_test = pickle.load(open("kiran_testing_inputs.p", 'rb'))

'''
A50 = pickle.load(open("A50.p", 'rb'))
'''

print 'finished loading'
'''
# use query function to test train/test accuracies 
# training accuracy
kiran50_correct1 = sum(is_right(A50, t[0], t[1], t[2], t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran50_correct5 = sum(topk_performance(A50, t[0], t[1], t[2], 5, t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran50_correct10 = sum(topk_performance(A50, t[0], t[1], t[2], 10, t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran50_correct20 = sum(topk_performance(A50, t[0], t[1], t[2], 20, t[3]) for t in kiran_test) / (0. + len(kiran_test))

print "-------------------------------------------------"
print "Training Accuracy After 50 Iterations (Compared to Arora's Word Vectors): "
print "Top 1 Accuracy Score: " + str(kiran50_correct1)
print "Top 5 Accuracy Score: " + str(kiran50_correct5)
print "Top 10 Accuracy Score: " + str(kiran50_correct10)
print "Top 20 Accuracy Score: " + str(kiran50_correct20)
'''




print 'finished loading'

# use query function to test train/test accuracies 
# training accuracy
kiran1000_correct1 = sum(is_right(A1000, t[0], t[1], t[2], t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran1000_correct5 = sum(topk_performance(A1000, t[0], t[1], t[2], 5, t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran1000_correct10 = sum(topk_performance(A1000, t[0], t[1], t[2], 10, t[3]) for t in kiran_test) / (0. + len(kiran_test))
kiran1000_correct20 = sum(topk_performance(A1000, t[0], t[1], t[2], 20, t[3]) for t in kiran_test) / (0. + len(kiran_test))

print "-------------------------------------------------"
print "Training Accuracy After 1000 Iterations (Compared to Arora's Word Vectors): "
print "Top 1 Accuracy Score: " + str(kiran1000_correct1)
print "Top 5 Accuracy Score: " + str(kiran1000_correct5)
print "Top 10 Accuracy Score: " + str(kiran1000_correct10)
print "Top 20 Accuracy Score: " + str(kiran1000_correct20)


# query where indices are presented instead of words
# different objective for arora
def query_top1_arora(A_mat, j1, j2, j3):
	v1 = A_mat[j1] 
	v2 = A_mat[j2]
	v3 = A_mat[j3]
	argmin = 0
	min_val = 0
	for i in range(0, len(arora_index_dict)):
		# need to restrict so max val in input is 1 or -1
		if (i != j1) and (i != j2) and (i != j3):
			v4 = A_mat[i]
			objective = norm(v1 - v2 - v3 + v4)**2
			if objective < min_val:
				min_val = objective
				argmin = i
	# return the best fit word and its objective value.
	return argmin, min_val

def is_right_arora(A_mat, j1, j2, j3, ans):
	argmin, min_val = query_top1_arora(A_mat, j1, j2, j3)
	if argmin == ans:
		return 1
	else:
		return 0

# query where indices are presented instead of words
# returns topk results in a set
def query_topk_arora(A_mat, j1, j2, j3, topk):
	v1 = A_mat[j1] 
	v2 = A_mat[j2]
	v3 = A_mat[j3]
	# we will negate the objective and find the max (to find the min of unnegated version)
	topk_argmaxes = []
	smallest_maxval = float('inf')
	for i in range(0, n):
		# need to restrict so max val in input is 1 or -1
		if (i != j1) and (i != j2) and (i != j3):
			v4 = A_mat[i]
			# negation
			objective = -1*norm(v1 - v2 - v3 + v4)**2
			if len(topk_argmaxes) < topk:
				hq.heappush(topk_argmaxes, (objective, i))
				if objective < smallest_maxval:
					smallest_maxval = objective
			elif objective > smallest_maxval:
				val, index = hq.heappop(topk_argmaxes)
				smallest_maxval = topk_argmaxes[0][0]
				if objective < smallest_maxval:
					smallest_maxval = objective
				hq.heappush(topk_argmaxes, (objective, i))
	# return the best fit word and its objective value.
	return set(map(lambda (a, b): b, topk_argmaxes))

# returns 1 if answer is in the top k answers
# k > 1
def topk_performance_arora(A_mat, j1, j2, j3, topk, ans):
	topk_argmins = query_topk_arora(A_mat, j1, j2, j3, topk)
	if ans in topk_argmins:
		return 1
	else:
		return 0

'''
# use query function to test Arora word vectors accuracy
# (was not trained on these vectors or anything, so we test on all of them)
arora_correct1 = sum(is_right_arora(arora_wvs, t[0], t[1], t[2], t[3]) for t in arora_test) / (0. + len(arora_test))
print 'finished 1'
arora_correct5 = sum(topk_performance_arora(arora_wvs, t[0], t[1], t[2], 5, t[3]) for t in arora_test) / (0. + len(arora_test))
print 'finished 2'
arora_correct10 = sum(topk_performance_arora(arora_wvs, t[0], t[1], t[2], 10, t[3]) for t in arora_test) / (0. + len(arora_test))
arora_correct20 = sum(topk_performance_arora(arora_wvs, t[0], t[1], t[2], 20, t[3]) for t in arora_test) / (0. + len(arora_test))

print "-------------------------------------------------"
print "Arora Accuracy: "
print "Top 1 Accuracy Score: " + str(arora_correct1)
print "Top 5 Accuracy Score: " + str(arora_correct5)
print "Top 10 Accuracy Score: " + str(arora_correct10)
print "Top 20 Accuracy Score: " + str(arora_correct20)
'''
