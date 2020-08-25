

#GATE pos english corpus downloaded from https://gate.ac.uk/wiki/twitter-postagger.html
#HMM model trained using NLTK
#https://www.nltk.org/_modules/nltk/tag/hmm.html
#Spanish data downloaded from https://www.clarin.si/repository/xmlui/handle/11356/1078?show=full
#http://nl.ijs.si/janes/wp-content/uploads/2016/09/A-Multilingual-Social-Media-Linguistic-Corpus.html

#import stanza
import nltk
from nltk.tag.hmm import HiddenMarkovModelTrainer,HiddenMarkovModelTagger

import glob


#file = "twitie-tagger/corpora/ark.oct27.agree"

#f=open(file)
#lines = [line.strip().split() for line in f]
#f.close()

#tokenized_docs = [[word.split("_")[-2:] for word in line if len(word)>1] for line in lines]
#tokenized_docs_tuples = [[tuple(word) for word in line] for line in tokenized_docs]

tokenized_docs = []
for file in glob.glob("twitie-tagger/corpora/*stanford"):
	print (file)
	f=open(file)
	lines = [line.strip().split() for line in f]
	f.close()

	tokenized_docs = tokenized_docs + [[word.split("_")[-2:] for word in line if len(word)>1] for line in lines]

for file in glob.glob("twitie-tagger/corpora/*agree"):
	print (file)
	f=open(file)
	lines = [line.strip().split() for line in f]
	f.close()

	tokenized_docs = tokenized_docs + [[word.split("_")[-2:] for word in line if len(word)>1] for line in lines]


tokenized_docs_tuples = [[tuple(word) for word in line] for line in tokenized_docs]

for sent in tokenized_docs_tuples:
	for word in sent:
		if len(word) != 2:
			print (word)

words = [word[0] for line in tokenized_docs for word in line]
wordsVocab = list(set(words))
states = [word[1] for line in tokenized_docs for word in line if len(word)>1]
statesVocab = list(set(states))

#HMMtrainer = HiddenMarkovModelTrainer(states=statesVocab,symbols=wordsVocab)

HMMtrainer = HiddenMarkovModelTrainer()
hmmmodel = HMMtrainer.train(tokenized_docs_tuples)


#print (hmmmodel.tag("wtf did u do ?".split()))
#sentence = "my home is burning".split()
#print (hmmmodel.tag(sentence))
