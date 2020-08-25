from spellchecker import SpellChecker
import re
import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz0"
c2i = {c:i for i,c in enumerate(alphabet)}
i2c = {i:c for i,c in enumerate(alphabet)}



spell = SpellChecker(language="es",distance=1)


def get_errors(sentencestr):

    changes = []
    wordsclean = re.sub("[^a-zA-Z\ ]","",sentencestr).lower().split(" ")
    misspelled = spell.unknown(wordsclean)
    for word in misspelled:
        suggestion = spell.correction(word)
                                          
        if (len(word) == len(suggestion)):
            changed_idx = [i for i,c in enumerate(word) if word[i]!=suggestion[i]]
            if len(changed_idx) >0:
#                the_change = word[changed_idx[0]]+"->"+suggestion[changed_idx[0]]
                the_change = (word[changed_idx[0]],suggestion[changed_idx[0]])
                #print (word,the_change)
                changes.append(the_change)
        elif  (len(word) > len(suggestion)):
            letter = word[-1]
            found=False
            for i,c in enumerate(suggestion):
                if found==False and word[i]!=suggestion[i]:
                    letter = word[i]
                    found=True
#            the_change = letter+"->0"
            the_change = (letter,"0")
            #print (word,the_change)
            changes.append(the_change)
        elif  (len(word) < len(suggestion)):
            letter = suggestion[-1]
            found=False
            for i,c in enumerate(word):
                if found==False and word[i]!=suggestion[i]:
                    letter = suggestion[i]
                    found=True
#            the_change = "0->"+letter
            the_change = ("0",letter)
            #print (word,the_change)
            changes.append(the_change)
    return changes

def get_error_vector(sentencestr):
    chgs = get_errors(sentencestr)
    vector = np.zeros(len(alphabet)**2)

    for chg in chgs:
        fromc=c2i.get(chg[0])
        toc=c2i.get(chg[1])
        if fromc is not None and toc is not None:
            print (chg, fromc,toc)
            position = c2i.get(chg[0]) * len(c2i) + c2i.get(chg[1])
            print (position)
            print ()
            vector[position] = vector[position] + 1
    return vector

def initialize_vector():
    return np.zeros(len(alphabet)**2)

sentence = '''I have a realy big probem maño'''
#print (get_errors(sentence))
#get_error_vector(sentence)
