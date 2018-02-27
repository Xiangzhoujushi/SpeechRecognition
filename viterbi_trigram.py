
# import Viterbi
from collections import Counter
import math
import sys 
import numpy as np

size = 1000

def build_probabilities(cond_count):
		  """
		  Given a conditional counter (count of x given y), return the corresponding log probabilities.

		  :param cond_count: (dict)
				Mapping of some reference key to counter object (EX: cc[tag][word] >> count)

		  :return cp: (dict)
				Mapping of reference key to conditional probabilities (EX: cp[tag][word] >> conditional probability)
		  """
		  cond_prob = {}
		  for outer, counter in cond_count.iteritems():
				total = sum(counter.values())
				# Note the use of log probability
				cond_prob[outer] = {inner: np.log(1.0 * icount / total) for inner, icount in counter.iteritems()}
		  return cond_prob

def main():

	tag_count_unigram = dict()
	tag_count_trigram = dict()
	tag_given_tag_counts = dict()
	word_given_tag_counts = dict()
	word_sequence = []
	actual_tags=[]

	with open ('./pos/pos_train.txt',"r") as infile:

		for line in infile:
		  # first tag is the start symbol
			lasttag2 ="<s>"
			lasttag= "<s>"
		  #
		  # split line into word/tag pairs
		  #
			for wordtag in line.rstrip().split(" "):
				if wordtag == "":
					 continue
				# note that you might have escaped slashes
				# 1\/2/CD means "1/2" "CD"
				# keep 1/2 as 1\/2 
				parts=wordtag.split("/")
				tag = parts.pop()
				word="/".join(parts)

				# add word to word sequence
				word_sequence.append(word)
				actual_tags.append(tag)
				#update the unigram
				if tag not in tag_count_unigram:
					tag_count_unigram[tag] = 0
				tag_count_unigram[tag] += 1
				# update counters
				if tag not in word_given_tag_counts:
					 word_given_tag_counts[tag]=Counter()
				if lasttag not in tag_given_tag_counts:
					 tag_given_tag_counts[lasttag]=Counter()
				word_given_tag_counts[tag][word]+=1
				tag_given_tag_counts[lasttag][tag]+=1


				pair = lasttag2,lasttag
				# print(pair)
				if pair not in tag_count_trigram:
					# pair = (lasttag,tag)
					tag_count_trigram[pair] = Counter()
				tag_count_trigram[pair][tag]+=1
				
				lasttag2 = lasttag
				lasttag = tag
				

				# if tag is '.':
				# 	lasttag='<s>'
				# 	lasttag2='<s>'

	# then process distribution
	# tag_tag_dist = deleted_interpolation(tag_count_unigram,tag_count_trigram,tag_given_tag_counts)
	# for key in tag_count_trigram:
	# 	print(key)

	tag_tag_dist = build_probabilities(tag_count_trigram)
	tag_pairs = tag_tag_dist.keys()
	tags = tag_count_unigram.keys()
	word_tag_dist = build_probabilities(word_given_tag_counts)
	word_enter = word
	#
	# size = 10
	trigram_viterbi(tag_tag_dist,tags, word_tag_dist,tag_pairs,word_sequence[0:size],actual_tags[0:size])		



def deleted_interpolation(tag_count_unigram,tags,tag_count_trigram,tag_count_bigram):
	a1 = 0
	a2 = 0
	a3 = 0
	count = len(tag_count_unigram)
	tag_tags_dist = tag_count_trigram.copy()
	for pair,value in tag_count_trigram:
		tag1,tag2 = pair
		for tag3 in value:
			if (tag_count_trigram[pair][tag3]-1)/(tag_count_bigram[tag1][tag2]-1): 
				a3+= tag_count_trigram[pair][tag3]
			elif (tag_count_bigram[tag2][tag3]-1)/(tag_count_unigram[tag2]-1):
				a2+= tag_count_trigram[pair][tag3]
			elif (tag_count_unigram[tag3]-1)/(count-1): 
				a1+=tag_tags_dist[pair][tag3]
			tag_tags_dist[pair][tag3] = a3*(tag_tags_trigram[pair][tag3]/tag_count_bigram[tag1][tag2])
			+a2*tag_count_bigram[tag2]/tag_tags_trigram[tag2][tag3]+a1+tag_count_unigram[tag3]/count
	return tag_tags_dist

# transition_probability is the map, 
def trigram_viterbi(tag_tag_dist,tags,word_tag_dist,tag_pairs,word_sequence,actual_tags):
	# given two distribution maps, just implement the viterbi algorithm
	# list of tag given tag pairs
	# distribution of word given tag
	# word_sequence = [(word,index) ]: all words, index used as specified states
	# tag_pairs, used as the hidden tags (tag_i-2,tag_i-1)
	

	# word_count = len(word_sequence)
	# print word_count

	# tag_count = len(actual_tags)

	# print tag_count

	# print(tag_pairs)
	m = len(word_sequence) # number of states
	n = len(tag_pairs) # number of tag_pair

	optimal = [[-10000 for x in range(n)] for y in range(m)]
	previous = [[-1 for x in range(n)] for y in range(m)]


	#previous track the tag index in the previous states
	pair = '<s>','<s>'
	# print() tag_pairs[pair]

	for j in range(len(tag_pairs)):
		# it it is in the pair
		
		tag1 = tag_pairs[j][1]
		if tag_tag_dist[pair].has_key(tag1):
			# tag_tag_dist[pair][tag_pairs[j]]
			try:
				prob_tag_word = word_tag_dist[tag1][word_sequence[0]]
				transition_prob = tag_tag_dist[pair][tag1]
				# print(prob_tag_word + transition_prob)
				prob = prob_tag_word + transition_prob
				previous[0][j] = -1				
				optimal[0][j] = prob
				# print('fuck+ ', str(prob))
			except KeyError:
				# optimal[0][j] = 1
				# optimal[0][j] = -10000
				previous[0][j] = -1
				# print("j "+str(j))


	for i in range(1,m):
		for j in range(len(tag_pairs)):
			tag_current = tag_pairs[j][1]
			tag_prev = tag_pairs[j][0]
			for k in range(len(tag_pairs)):
			# if the tags are matching
				if (optimal[i-1][k]>-10000) and (tag_pairs[j][0] == tag_pairs[k][1]):
				#previous tag, log use all multiplication
					try:
						prob_tag_word = word_tag_dist[tag_pairs[j][1]][word_sequence[i]]
						transition_prob = tag_tag_dist[tag_pairs[k]][tag_pairs[j][1]]
						prob = prob_tag_word + transition_prob

						if optimal[i-1][k]+ prob > optimal[i][j]:
							optimal[i][j] = optimal[i-1][k]+ prob
							# update the argument
							previous[i][j]= k
							# print (i,k,j)	
					except KeyError:
						previous[i][j] = k
						# do nothing
	a = m-1
	# b is the max n
	b = 0
	for i in range(1,len(optimal[a])):
		if optimal[a][b]<optimal[a][i]:
			b =i

	li= []

	# print("best one", b)
	# li.insert(0,tag_pairs[a][b])
#back track to get the previous tag
	while previous[a][b]>=0:
		# print(li)	
		li.insert(0,tag_pairs[b][1])
		b = previous[a][b]
		a = a-1
	# li.insert(0,tag_pairs[a][1])
	
	li.insert(0,tag_pairs[b][0])

	

	# print(word_count)
	# print(li)
	# # print(len(word_sequence))
	# print(actual_tags)

	correct_tags = 0
	correct = sum([li[i] == actual_tag for i, actual_tag in enumerate(actual_tags)])
	correct_tags += correct
	total_tags = len(actual_tags)

	print ('Number of words used:', total_tags)
	print('Accuracy: {0} / {1} = {2:0.1f}%'.format(correct_tags, total_tags, 100.0 * correct_tags / total_tags))

# def main():



main()









