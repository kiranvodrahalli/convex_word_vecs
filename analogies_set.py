import nltk
from nltk.corpus import wordnet as wn
import pickle
#import enchant

# size = 32081; if use all words in THIS vocab: 
#                 if m = 10 -> sample complexity will be upperbounded by 320081.
#                 if m = 100 -> sample complexity will be a lot worse.....
# is a set()
vocab = pickle.load(open("vocab.p", "rb"))
#eng = enchant.Dict("en")

# this is a test of principle basically: can we learn word vectors which learn an actually defined
# linguistic analogy? 
# first task: try to learn the "antonym" relationship with this analogy framework. 
# i.e. the word vectors we get should be able to tell if two words are antonyms. That is it!
# the reduction from the analogy training set up as follows: we simply let w1, w2 be a pair of antonyms
# then w3 is the word we want to find the antonym for. then we take our argmin to find w4 which is its antonym.
# this should work for any pair of antonyms w1, w2 assuming we train across all antonyms. 
# this problem should be easier to learn than the analogies for now (since we would have to generate 
# a lot of analogies, which is a nontrivial task)

# implement this for paper; future work: actually implement on analogy dataset + compare to arora. 

# analogy will be a 4-tuple: (w1, w2, w3, w4) means w1:w2 :: w3:w4
# threshold is for early stopping, when we have enough analogies.
def build_antonym_analogies(threshold):
	vocab_used = set()
	antonym_analogies = set()
	for w1 in vocab:
		for w3 in vocab:
			if w1 != w3:
				# we are building a bunch of antonym pairs to learn word vectors which encode antonyms.
				w1_synsets = wn.synsets(w1)
				w3_synsets = wn.synsets(w3)
				w1_lemmas, w3_lemmas = [], []
				if len(w1_synsets) > 0:
					w1_lemmas = w1_synsets[0].lemmas()
				if len(w3_synsets) > 0:
					w3_lemmas = w3_synsets[0].lemmas()
				for l1 in w1_lemmas:
					a1s = l1.antonyms()
					for l3 in w3_lemmas:
						a3s = l3.antonyms()
						for a1 in a1s:
							w2 = a1.key().split("%")[0]
							for a3 in a3s:
								w4 = a3.key().split("%")[0]
								# we want to retain direction w1 is antonym to w2 => w3 is antonym to w4. 
								antonym_analogies.add((w1, w2, w3, w4))
								antonym_analogies.add((w2, w1, w4, w3))
								antonym_analogies.add((w3, w4, w1, w2))
								antonym_analogies.add((w4, w3, w2, w1))
								vocab_used.add(w1)
								vocab_used.add(w2)
								vocab_used.add(w3)
								vocab_used.add(w4)
								print "#" + str(len(antonym_analogies)) + ": " + str((w1, w2, w3, w4))
								if len(antonym_analogies) > threshold:
									return antonym_analogies, vocab_used
	return antonym_analogies, vocab_used

# in fact, perhaps a better mapping to encode the idea of antonym analogy: 
# ('good', 'bad', w1, w2), where w1 and w2 are antonyms. We just stick with one query at first -> then this
# effectively becomes a supervised way to learn word vectors for antonyms.. though our approach is
# actually overkill. 

# NEED TO GENERATE THINGS THAT ARE NOT ANTONYMS NOW!! - to double check... actually, this is not such a problem.
# both synonyms, and unrelated words! - not such a problem. 

# we divide up our data into training and non-training. we need every word in vocabulary to be in both training
# and not training - otherwise we will not have trained that word at all! Remember only word 3 matters here, 
# because we are just trying to find an antonym for w3. 


# build TRAINING/TESTING SETS
# cross-validation 80% train / 20% test. 














