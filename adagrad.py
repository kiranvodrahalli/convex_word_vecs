import numpy as np 
import pickle
from math import sqrt
from numpy.linalg import norm
from numpy.polynomial import Polynomial 
from numpy.polynomial.polynomial import polymul
from numpy.polynomial.polynomial import polyadd 
from copy import deepcopy


# vocab set
vocab3 = pickle.load(open("antonyms_300002_vocab.p", "rb"))
# one of the dimensions of matrix A (k x n)
n = len(vocab3)
# we choose k = 10
k = 5 # explained in paper

'''
# antonym 4-tuple set
antonyms3 = pickle.load(open("antonyms_300002.p", "rb")) 

# code to generate the pickled files

# build word - index dictionary 
word_dict = dict(zip([w for w in vocab3], range(len(vocab3))))
# build index-word dictionary
index_dict = dict(zip(range(len(vocab3)),[w for w in vocab3]))

# convert 4-tuples into indices. 
input_list = []
for a in antonyms3:
	new_tuple = (word_dict[a[0]], word_dict[a[1]], word_dict[a[2]], word_dict[a[3]])
	input_list.append(new_tuple)
'''

# NOTE: In this code, we instead transform A into a_t column wise. This way, a_t becomes a concatenation
#       of the word vectors. 

# loss function: a_t is A, x_t is the input of the word in index form (in [0, n-1])
def loss(a_t, x_t):
	print "Calculate loss "
	print x_t
	i1 = x_t[0]
	i2 = x_t[1]
	i3 = x_t[2]
	y = x_t[3] # fourth input in training model is the answer index
	max_loss = 0
	argmax = 0
	for j in range(0, n):
		# no duplicates in encoding
		if j != y and j != i1 and j != i2 and j != i3:
			j_loss = norm(-a_t[i1*k: (i1+1)*k] + a_t[i2*k: (i2+1)*k] + a_t[i3*k: (i3+1)*k] + a_t[j*k: (j+1)*k])**2	
			if j_loss > max_loss:
				max_loss = j_loss
				argmax = j
	return max_loss, argmax


# gradient of loss function
# j is winning affected index after loss is calculated (argmax output from loss(a_t, x_t))
def grad(a_t, x_t, j):
	print "Calculate grad"
	i1 = x_t[0]
	i2 = x_t[1]
	i3 = x_t[2]
	# gradient vector
	g = np.zeros(n*k)
	g[i1*k:(i1+1)*k] = -2*(-a_t[i1*k: (i1+1)*k] + a_t[i2*k: (i2+1)*k] + a_t[i3*k: (i3+1)*k] + a_t[j*k: (j+1)*k])
	g[i2*k: (i2+1)*k] = 2*(-a_t[i1*k: (i1+1)*k] + a_t[i2*k: (i2+1)*k] + a_t[i3*k: (i3+1)*k] + a_t[j*k: (j+1)*k])
	g[i3*k: (i3+1)*k] = 2*(-a_t[i1*k: (i1+1)*k] + a_t[i2*k: (i2+1)*k] + a_t[i3*k: (i3+1)*k] + a_t[j*k: (j+1)*k])
	g[j*k: (j+1)*k] = 2*(-a_t[i1*k: (i1+1)*k] + a_t[i2*k: (i2+1)*k] + a_t[i3*k: (i3+1)*k] + a_t[j*k: (j+1)*k])
	cnt = 0
	for i in range(0, n*k):
		if g[i] > 0:
			cnt+=1
	print cnt
	return g


def vec_sq(v):
	return map(lambda z:z**2, v)
# projecting point b_t onto K: R^(kn) according to G_t, with each column vector having norm 1
# need to take k-slices of b_t, and project each k-slice separately according to the 
# corresponding k-slice of G_t onto the unit k-ball. (recall that G_t is diagonal matrix expressed as array)
# TRANSLATED INTO PYTHON FROM http://www.mathworks.com/matlabcentral/fileexchange/27711-euclidian-projection-on-ellipsoid-and-conic
# ORIGINAL CODE BY BRUNO LUONG: CONVERTED BY KIRAN VODRAHALLI
def project(b_t, G_t):
	print "Project"
	# first loop over starting k-slice indices.
	# for each of these, build G_t[:, a:a + k] = s (what defines the ellipsoid in R^k)
	# get k-slice of b_t too: b_t[:, a:a + k] = d (the point in $R^k$). (go over columns, remember)
	# go through polynomial solving
	# y is the output. 
	# ADD EXPLANATION SECTION FOR THIS IN THE PAPER. 
	# ACTUALLY CAN TOTALLY IMPLEMENT THIS IN PYTHON - DO nOT NEED MATLAB AFTER ALL. 
	projected_pt = np.zeros(np.shape(b_t))
	S_t = np.array(map(lambda z: sqrt(z), G_t)) # x^t G_t^(1/2) x is our ellipse
	# for rescaling: G^(-1/4)
	G_quarterth_t = np.array(map(lambda z: 1./sqrt(z), S_t))
	for m in range(0, n):
		# k-slices
		d = b_t[m*k:(m+1)*k]
		s = S_t[m*k:(m+1)*k]
		rs = G_quarterth_t[m*k:(m+1)*k] #rescale
		pd = map(lambda (a, b, c): Polynomial(np.array([a, b, c])), zip(vec_sq(s), 2*s, np.ones(len(s))))
		a = np.multiply(s, vec_sq(d))
		#- Pc is product of all polynomials: 
		#- PROD [ pd(j,:) for j=1,2, ..., n ]
		pc = reduce(lambda c1, c2: polymul(c1, c2), pd)[0]
		#- Pi is an array polynomials of product but leaving-one-out.
		#- Pi(i,:) = PROD [ pd(j,:) for j not i ]
		pi = []
		for i in range(len(pd)):
			new_pd = deepcopy(pd)
			del new_pd[i]
			pcurr = reduce(lambda c1, c2: polymul(c1, c2), new_pd)
			pi.append(pcurr)
		# multiply by a and add up coefficients for each term
		xx = map(lambda (c, d): np.multiply(c, d), zip(a, pi))
		P = reduce(lambda c1, c2: polyadd(c1, c2), xx)[0]
		# reverse coefficient order for finding the correct roots
		coefs = np.r_[0, 0, P.coef] - pc.coef
		rev_coef = np.fliplr([coefs])[0]
		P = Polynomial(rev_coef)
		# solve for roots
		lambdas = P.roots()
		# filter out non-real roots
		real_roots = map(lambda z: z.real, filter(lambda z: z.imag == 0., lambdas))
		#Compute: y = d / (lambda*s + 1)
		min_point = np.zeros(k)
		min_dist = float('inf')
		for root in real_roots:
			ls = root*s + 1
			y = np.multiply(d, np.array(map(lambda z: 1./z, ls)))
			curr_dist = norm(y-d)
			if curr_dist < min_dist:
				min_dist = curr_dist
				min_point = y
		# rescale
		min_point = np.multiply(rs, min_point)
		# check that projection is correct!
		if norm(min_point) > 1.0000001:
			print norm(min_point)
		# paste together
		if m == 0:
			projected_pt = min_point
		else:
			projected_pt = np.r_[projected_pt, min_point]
	return projected_pt

# eta = learning rate
eta = sqrt(2)

# delta = G_1 init
delta = 1

# gets a random vector on the unit k-ball
def get_rand_unit(z):
	v = np.random.rand(z)
	return v/norm(v)

# update step in adagrad
def ada_update(a_t, x_t, G_t):
	print "next instance"
	# l = loss, j = index of loss
	l, j = loss(a_t, x_t)
	g_t = grad(a_t, x_t, j) 
	print 'norm of gradient: ' + str(norm(g_t))
	G_t = G_t + np.array(map(lambda z: z**2., g_t))
	# separate learning rates
	G_divt = np.array(map(lambda z: 1./sqrt(z), G_t))
	b_t = a_t - eta*np.multiply(G_divt, g_t)
	# project b_t onto K to get a_{t + 1}
	a_tp1 = project(b_t, G_t)
	return a_tp1, G_t

# something that may be interesting to try: repeat input several times in a row. 
# (should be able to optimize for this): this way, you get mileage out of the max function. 
# (decrease several things that are not the max for that particular problem) - addresses
# the problem of learning the training set. 
# yes, something is fishy with this whole set up I think... you need certain KINDS of data
# in order to learn, not just an amount. if you keep getting data from the same area (i.e.
# it is not uniformly distributed...this is a problem). 

# another thing: distance 2 --> 2^ = 4: # of column vectors being changed, each has l2 norm 1. 
# so roughly distance of sqrt(4) between A_t and A_t+1

# train is training set: list of 4-tuples like input_list
def adagrad(train):
	# randomly initialize: join n k-dimensional unit vectors row-wise.
	a_t = np.zeros(n*k)
	for i in range(0, n):
		if i == 0:
			a_t = get_rand_unit(k)
		else:
			a_t = np.r_[a_t, get_rand_unit(k)]
	# delta = 1
	G_t = np.ones(n*k) # for keeping track of sum of gradient squares (diagonal matrix)
	avg_a = a_t
	update_diff_train = []
	for x_t in train:
		a_tp1, G_t = ada_update(a_t, x_t, G_t)
		diff = norm(a_t - a_tp1)
		update_diff_train.append(diff)
		print "Difference between old and new a_t: " + str(diff)
		# update!
		a_t = a_tp1
		# update average!
		avg_a = avg_a + a_tp1
		print "------------------------------------------"
	T = len(train) + 0.0
	a = (1/T)*avg_a
	A = np.zeros((n, k))
	for m in range(0, n):
		A[m] = a[m*k:(m+1)*k]
	# rows are word vectors
	return A, update_diff_train
