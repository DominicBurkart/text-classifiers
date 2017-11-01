"""
Naive Bayes Classifier that considers each alphanumeric word as a distinct feature,
with no special treatment to specific word locations (e.g. first word) etc.

Removes emoji text that has been transcribed into unicode (e.g. \x99).

Prints the accuracy of the classifer and 100 most informative features to the console.

Saves the trained classifier in the working directory.

Dominic Burkart dominicburkart@nyu.edu
"""

import nltk
import random
from pandas import *
import re
import pickle

def getFeatureSet(twoples):
    """
    @param twoples: the text / categorization from which the word featureset should be extracted.
    The format should look like [("Text with words", "categorization of text"), ("another text", "same/another category")]
    """
    out = [] #list of dictionary/label tuples
    for t in twoples:
        d = dict()
        text = t[0]
        text = re.sub(r'\\x[A-Fa-f0-9]+', ' ', text) #removes characters transcribed from unicode to ascii (e.g. emoji)
        text = re.sub("[^a-zA-Z\d\s']", ' ', text) #only keeps alphanumeric characters
        words = text.lower().split()
        for w in words:
            d[w] = 1
        out.append((d, t[1])) #appends a tuple with the dictionary object and the category to the out list
    return out #we have our feature set! send it out.

fr = concat([read_table('clinton_fb_post_comments_ordered.csv'), read_table('trump_fb_post_comments_ordered.csv')])

full = [(x, y) for x,y in zip(list(fr.post), list(fr.post_author))]
random.seed(405968) #explicitly set for reproducibility
random.shuffle(full) #shuffles all documents to avoid order effects
train = getFeatureSet(full[:int(len(full)*0.8)]) #gets features from half of the documents
test = getFeatureSet(full[int(len(full)*0.8):]) #gets features from the remaining 20%

cl = nltk.NaiveBayesClassifier.train(train)
print("Classifier accuracy: "+str(nltk.classify.accuracy(cl, test)))
cl.show_most_informative_features(n = 100)

n_bytes = 2**31 #the pickle package is slightly broken in python 3.4 and requires these params for saving large files.
max_bytes = 2**31 - 1
bytes_out = pickle.dumps(cl)
with open('equal-word-features-bayes.obj', 'wb') as f_out:
    for idx in range(0, n_bytes, max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])
