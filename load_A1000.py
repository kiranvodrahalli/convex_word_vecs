import numpy as np
import pickle

mat_array = []

f = open("A1000.txt", "rb")

for line in f:
	l1 = line.translate(None, "[]\n")
	l2 = l1.split(" ")
	l3 = [c for c in l2 if c != '']
	print l3
	l4 = np.array(map(lambda z: float(z), l3))
	mat_array.append(l4)
A = np.array(mat_array)
pickle.dump(A, open("A1000.p", 'wb'))