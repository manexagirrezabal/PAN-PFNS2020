
#General imports
import glob
import numpy as np
import os
import sys

#Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#Validation packages
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

#Library to open XML files
from xml.etree import ElementTree as ET

#Setting some variables
#Get home directory
HOME = os.environ['HOME']

#Set the directory where we saved the corpus of fake news spreaders
#DIREC = HOME+"/corpusak/pan20-author-profiling-training-2020-02-23/"

#Set the language
#LANG  = "en"
LANG = sys.argv[1]
DIREC = sys.argv[2]
OUTPUTDIR = sys.argv[3]

def count_upper_lower(st):
    upp = 0
    low = 0
    for c in st:
        if c.isupper():
            upp = upp + 1
        elif c.islower():
            low = low + 1
    return (upp,low)

if LANG == "en":
    from embeddings.embeddingeng import wembs as wembsesp
elif LANG == "es":
    from embeddings.embeddingsesp import wembs as wembsesp
else:
    print ("ERROR EMB")
    exit()

if LANG == "en":
    from pos import hmmmodel
elif LANG == "es":
    from poses import hmmmodel
else:
    print ("ERROR HMM")
    exit()
list_of_poss = hmmmodel._states

(wdict,nembs,vsize) = wembsesp

#simple test
def get_representation_tweets(F):
#    f=open(FILE)
#    l = f.read()
#    f.close()

    parsedtree = ET.parse(F)
    documents = parsedtree.iter("document")

    upsum = 0
    losum = 0
    texts = []
    origtexts = []
    for doc in documents:
        origtexts.append(doc.text)
        texts.append(doc.text.lower())
        up,lo = count_upper_lower(doc.text)
        upsum = upsum + up
        losum = losum + lo
    upper_ratio = upsum/(upsum+losum)

    lengths = [len(tweet) for tweet in texts]

#    print (lengths)

#    print (texts)

    sumembs = np.zeros((nembs))
    ntoks=0
    for text in texts:
        tokens = text.split(" ")
        for token in tokens:
            if token in wdict:
                sumembs = sumembs+wdict[token]
                ntoks = ntoks+1
        
    print ("Mean embeddings calculated")
    sumembs = sumembs/ntoks

    #It looks like from the results that the model doesn't
    #get better when using the standard deviation of the embeddings :-(
    #Therefore, I will just remove it.
    devembs = np.zeros((nembs))
    for text in texts:
        tokens = text.split(" ")
        for token in tokens:
            if token in wdict:
            	devembs = devembs + (np.abs(wdict[token]-sumembs))
    devembs = devembs/(ntoks-1)
    devembs = np.sqrt(devembs)
    print ("Stdev embeddings calculated")



    bag_of_pos = np.zeros(len(list_of_poss), dtype=np.int)
    for text in texts:
        tokens = text.split(" ")
        tagged_tokens = hmmmodel.tag(tokens)
        tags = [wt[1] for wt in tagged_tokens]
        for indtag,tag in enumerate(list_of_poss):
            tagfreq = tags.count(tag)
            bag_of_pos[indtag] = bag_of_pos[indtag] + tagfreq
    bag_of_pos = bag_of_pos/np.max(bag_of_pos)
    print ("Bag of pos calculated! Normalized from 0 to 1")

    return np.concatenate(([np.mean(lengths), np.std(lengths),upper_ratio],sumembs,devembs,bag_of_pos))



#GT    = DIREC+LANG+"/truth.txt"
#true_values = {}
#f=open(GT)
#for line in f:
#    linev = line.strip().split(":::")
#    true_values[linev[0]] = linev[1]
#f.close()

X = []
y = []

users = []
for FILE in glob.glob(DIREC+"/"+LANG+"/*.xml"):
    #The split command below gets just the file name,
    #without the whole address. The last slicing part [:-4]
    #removes .xml from the name, so that to get the user code
    USERCODE = FILE.split("/")[-1][:-4]

    #This function should return a vectorial representation of a user
    repr = get_representation_tweets (FILE)
    
    #We append the representation of the user to the X variable
    #and the class to the y vector
    X.append(repr)
    users.append(USERCODE)
#    y.append(true_values[USERCODE])

X = np.array(X)

import pickle
f=open("rf_classifier"+LANG+".pickle","rb")
rf_clf = pickle.load(f)
f.close()

os.mkdir(OUTPUTDIR+"/"+LANG)

y_pred = rf_clf.predict(X)

for idx,pred in enumerate(y_pred):
    print ("prediction no.",idx)
    usercode = users[idx]
    f=open(OUTPUTDIR+"/"+LANG+"/"+usercode+".xml", "w")
    print ("<author id=\""+usercode+"\" lang=\""+LANG+"\" type=\""+str(pred)+"\"/>",file=f)
    f.close()
