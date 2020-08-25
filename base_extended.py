
#General imports
import glob
import numpy as np
import os

#Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

#Validation packages
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

#Library to open XML files
from xml.etree import ElementTree as ET

#Setting some variables
#Get home directory
HOME = os.environ['HOME']

#Set the directory where we saved the corpus of fake news spreaders
DIREC = HOME+"/corpusak/pan20-author-profiling-training-2020-02-23/"

#Set the language
LANG  = "en/"


#simple test
def get_representation_tweets(F):
#	f=open(FILE)
#	l = f.read()
#	f.close()

	parsedtree = ET.parse(F)
	documents = parsedtree.iter("document")

	texts = []
	for doc in documents:
		texts.append(doc.text)

	lengths = [len(tweet) for tweet in texts]

#	print (lengths)

	return [np.mean(lengths), np.std(lengths)]



GT    = DIREC+LANG+"/truth.txt"
true_values = {}
f=open(GT)
for line in f:
	linev = line.strip().split(":::")
	true_values[linev[0]] = linev[1]
f.close()

X = []
y = []

for FILE in glob.glob(DIREC+LANG+"*.xml"):
    #The split command below gets just the file name,
    #without the whole address. The last slicing part [:-4]
    #removes .xml from the name, so that to get the user code
	USERCODE = FILE.split("/")[-1][:-4]

    #This function should return a vectorial representation of a user
	repr = get_representation_tweets (FILE)
	
    #We append the representation of the user to the X variable
    #and the class to the y vector
	X.append(repr)
	y.append(true_values[USERCODE])

X = np.array(X)
y = np.array(y)

skf = StratifiedKFold(n_splits=5)

evaluation_scores = [("accuracy",accuracy_score),("confusion",confusion_matrix)]

for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print (X_train.shape,y_train.shape)
	print (X_test.shape,y_test.shape)
	print ()
	dummy_clf = DummyClassifier(strategy="most_frequent")
	dummy_clf.fit(X_train, y_train)
	y_pred_dummy = dummy_clf.predict(X_test)
	
	for (evalname,evaluator) in evaluation_scores:
		print ("DUMMY "+evalname+":", evaluator(y_test,y_pred_dummy))

	lr_clf    = LogisticRegression(solver="lbfgs")
	lr_clf.fit(X_train, y_train)
	y_pred_lr = lr_clf.predict(X_test)
	for evalname,evaluator in evaluation_scores:
		print ("Logistic regression "+evalname+":", evaluator(y_test,y_pred_lr))



