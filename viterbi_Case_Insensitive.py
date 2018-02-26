from collections import Counter
from os.path import join
import numpy as np
import operator


# from patrick.viterbi import Viterbi; v = Viterbi()
class Viterbi(object):

    def __init__(self, folder='/vagrant/Lab2/SpeechRecognition/'):
        self.folder = folder
        self.neginf = -1000.0  # approximate negative infinity log probability (P=0)
        self.small = -7.0  # approximate very small probability (P=0.0000001)

    @staticmethod
    def _build_probabilities(cond_count):
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

    def read_train(self, filename, split_multitag):
        """
        Read the conditional counts from a labeled sentence file.

        :param filename: (str)
        :param split_multitag: (bool)

        :return:
        """
        filename = join(self.folder, filename)
        tag_given_tag_counts = dict()
        word_given_tag_counts = dict()
        unique_words = set()
        with open(filename, "r") as infile:
            # Each line in the file is a sentance
            for line in infile:
                if split_multitag:
                    # *** SOLUTION 1: plit multi tags into parts and count each separately ***
                    priortags = ['<s>']  # start symbol begins each sentence (simplifies algorithm)
                    # split into word/tag pairs (wordtag) and then into words and tags (parts)
                    for wordtag in line.rstrip().split(" "):
                        if wordtag == "":
                            continue
                        # Split on forward slash, but there may be escaped slashes in the word portion
                        #   EX: 1\/2/CD means "1/2" "CD"
                        parts = wordtag.split("/")
                        tags = parts.pop().split("")
                        word = "/".join(parts).lower()
                        unique_words.add(word)
                        # The tag maybe be multiple tags joined by "|" >> update counts for each tag
                        for tag in tags:
                            for lasttag in priortags:
                                if tag not in word_given_tag_counts:
                                    word_given_tag_counts[tag] = Counter()
                                if lasttag not in tag_given_tag_counts:
                                    tag_given_tag_counts[lasttag] = Counter()
                                word_given_tag_counts[tag][word] += 1
                                tag_given_tag_counts[lasttag][tag] += 1
                                priortags = tags
                else:
                    # *** SOLUTION 2: treat multi-tags as their own tags (EX: VBP|VB, NN|CD)
                    lasttag = '<s>'  # start symbol begins each sentence (simplifies algorithm)
                    # split into word/tag pairs (wordtag) and then into words and tags (parts)
                    for wordtag in line.rstrip().split(" "):
                        if wordtag == "":
                            continue
                        # Split on forward slash, but there may be escaped slashes in the word portion
                        #   EX: 1\/2/CD means "1/2" "CD"
                        parts = wordtag.split("/")
                        tag = parts.pop()
                        word = "/".join(parts).lower()
                        unique_words.add(word)
                        # The tag maybe be multiple tags joined by "|" >> update counts for each tag
                        if tag not in word_given_tag_counts:
                            word_given_tag_counts[tag] = Counter()
                        if lasttag not in tag_given_tag_counts:
                            tag_given_tag_counts[lasttag] = Counter()
                        word_given_tag_counts[tag][word] += 1
                        tag_given_tag_counts[lasttag][tag] += 1
                        lasttag = tag

        return word_given_tag_counts, tag_given_tag_counts, unique_words

    def read_test(self, filename):
        """
        Read the test data and split into sentences, tracking words and tags separately.

        :return results: (tuple)
            sentence_words: (list of list)
                The words for each sentence
            sentence_tags: (list of list)
                The tags for each word

            These should align, such that the index of a word aligns with the corresponding tag
        """
        filename = join(self.folder, filename)
        sentence_words = []
        sentence_tags = []
        with open(filename, "r") as infile:
            # Each line in the file is a sentance
            for line in infile:
                words = []
                tags = []
                # Split into word/tag pairs (wordtag) and then into words and tags (parts)
                for wordtag in line.rstrip().split(" "):
                    if wordtag == "":
                        continue
                    # Split on forward slash, but there may be escaped slashes in the word portion
                    #   EX: 1\/2/CD means "1/2" "CD"
                    parts = wordtag.split("/")
                    tags.append(parts.pop())
                    words.append("/".join(parts))
                sentence_tags.append(tags)
                sentence_words.append(words)

        return sentence_words, sentence_tags

    def run(self, train_file='pos/pos_train.txt', test_file='pos/pos_test.txt', split_multitag=False, debug=False):
        """
        Run the Viterbi algorithm on a series of sentences to predict tags.

        :param train_file:
        :param test_file:
        :param split_multitag:
        :param debug:

        :return:
        """
        # Load training data and compute conditional probabilities
        word_given_tag_counts, tag_given_tag_counts, unique_words = self.read_train(train_file, split_multitag)
        word_given_tag_prob = self._build_probabilities(word_given_tag_counts)
        tag_given_tag_prob = self._build_probabilities(tag_given_tag_counts)
        # Given a probability to the most popular tag for unknown words
        counts_by_tag = {tag: sum(tagcounts.values()) for tag, tagcounts in tag_given_tag_counts.iteritems()}
        most_popular_tag = max(counts_by_tag.iteritems(), key=operator.itemgetter(1))[0]
        word_given_tag_prob[most_popular_tag]['<UNK>'] = self.small
        # Each unique tag is a possible state (lattice columns)
        states = tag_given_tag_counts.keys()  # includes <s>
        istart = states.index('<s>')
        # Load test sentences
        test_sentence_words, actual_sentence_tags = self.read_test(test_file)
        total_tags = 0
        correct_tags = 0
        # Initialize the lattice for each sentence
        total_sentences = len(test_sentence_words)
        for isentence, sentence in enumerate(test_sentence_words):
            if isentence % 50 == 0:
                print('Processing sentence {0} of {1}'.format(isentence, total_sentences))
            backpointer = []  # row index of prior state with maximal probability at each state
            word_count = len(sentence)
            # Transpose of normal: rows=sentences, cols=states (simplifies iteration)
            #   Add one to rows for the start symbol
            viterbi = np.zeros([word_count + 1, len(states)])
            # Initialize probability of the start state to 1 (log probability 0), otherwise approx negative infinity
            viterbi[0] = [(0.0 if i == istart else self.neginf) for i, tag in enumerate(states)]
            # Iterate over each state for each possible state for each word in the sentence
            iword = 1  # start from one to allow for start symbol at 0
            for word in sentence:
		word = word.lower()
                if word not in unique_words:
                    word = '<UNK>'
                backpointer_row = []  # backpointers for each state at this step
                for istate, tag in enumerate(states):
                    # Probability of each possible incoming path = vi + aij + bj  (add log probabilities)
                    #   vi = viterbi probability of prior state
                    #   aij = transition probability from prior state to this state (tag given tag)
                    #   bj = observation probability of word in this state (word given tag)
                    vi = viterbi[iword - 1]
                    aij = [tag_given_tag_prob[prior_tag].get(tag, self.neginf) for prior_tag in states]
                    try:
                        bj = word_given_tag_prob[tag].get(word, self.neginf)  # same for all incoming paths
                    except KeyError:
                        # Happens for start tag outside start state >> zero probability
                        bj = self.neginf
                    incoming = map(operator.add, vi, aij)
                    ibest = np.argmax(incoming)
                    bestprob = incoming[ibest] + bj
                    # Record the winning probability for this state and the pointer to corresponding previous state
                    backpointer_row.append(ibest)
                    viterbi[iword][istate] = bestprob
                iword += 1
                backpointer.append(backpointer_row)

            # Choose the end state with the largest probability
            ibest = np.argmax(viterbi[-1])
            bestprob = viterbi[-1][ibest]
            bestpath = [states[ibest]]
            # Trace backwards along best path
            iword = word_count - 1
            prevstep = ibest
            while iword > 0:
                prevstep = backpointer[iword][prevstep]
                bestpath.append(states[prevstep])
                iword -= 1
            bestpath.reverse()  # account for backtracing
            actual_tags = actual_sentence_tags[isentence]

            # Calculate accuracy
            correct = sum([bestpath[i] == actual_tag for i, actual_tag in enumerate(actual_tags)])
            correct_tags += correct
            total_tags += word_count

            if debug:
                # DEBUG: probability, sentence, predicted and actual tags
                print('# SENTENCE: {0}'.format(isentence))
                print('most probable: {0}'.format(np.exp(bestprob)))
                print('Sentence: {0}'.format(sentence))
                print('Predicted: {0}'.format(bestpath))
                print('Actual: {0}'.format(actual_tags))
                print('Accuracy: {0} / {1} = {2:0.1f}%'.format(correct, word_count, 100.0 * correct / word_count))
                if isentence > 3:
                    break

        # Calculate Overall Accuracy
        print('Accuracy: {0} / {1} = {2:0.1f}%'.format(correct_tags, total_tags, 100.0 * correct_tags / total_tags))
        # Accuracy = 53937 / 56824 = 94.9%
        print('Done!')


v = Viterbi()
v.run()
