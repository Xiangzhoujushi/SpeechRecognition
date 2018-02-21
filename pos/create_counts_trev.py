from collections import Counter

# this is an example of how to parse the POS tag file and get counts
# needed for a bigram tagger 

tag_given_tag_counts=dict()
word_given_tag_counts=dict()

with open ("pos_train.txt","r") as infile:
    for line in infile:
        #
        # first tag is the start symbol
        lasttag="<s>"
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
            tag=parts.pop()
            word="/".join(parts)
            #
            # update counters
            if tag not in word_given_tag_counts:
                word_given_tag_counts[tag]=Counter()
            if lasttag not in tag_given_tag_counts:
                tag_given_tag_counts[lasttag]=Counter()
            word_given_tag_counts[tag][word]+=1
            tag_given_tag_counts[lasttag][tag]+=1
            lasttag=tag

# examples
pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR","JJS", "LS", "MD","NN","NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "$", "#", "(", ")", ",", ".", ":"]

for pos1 in pos_tags:
	for pos2 in pos_tags:
		print ('[{}][{}] = '+str(tag_given_tag_counts[pos1][pos2])).format(pos1, pos2)
print ("count[NN][VB] = "+str(tag_given_tag_counts["NN"]["VB"]))
print ("count[NN][dog] = "+str(word_given_tag_counts["NN"]["dog"]))

            
