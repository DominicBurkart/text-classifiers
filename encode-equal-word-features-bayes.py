"""
Used for running the equal word features classifer (created by train-word-features.py).

USAGE:

1.) make sure that the current directory is the folder that contains this
file and the other bayes encoding files in this suite.

2.) enter the command "python3 encode-equal-word-features-bayes.py path/to/input.tsv",
where path/to/input.tsv is replaced with the path to the text to be encoded.
If the filepath

Make sure that the TSV with the text to be encoded has a "message" column – that's
what this program uses.



Dominic Burkart dominicburkart@nyu.edu
"""
from pandas import *
import sys
import os
import pickle
import re

def getDict(text):
    """
    Takes a given string and makes it legible to the classifier.

    @param: the text which should be reformatted as a dictionary.
    """
    d = dict()
    text = re.sub(r'\\x[A-Fa-f0-9]+', ' ', text) #removes characters transcribed from unicode to ascii (e.g. emoji)
    text = re.sub("[^a-zA-Z\d\s']", ' ', text) #only keeps alphanumeric characters
    words = text.lower().split()
    for w in words:
        d[w] = 1
    return d

def classify(text):
    """
    Returns the relative probability that a given string would appear on
    Hillary Clinton's twitter (versus on Donald Trump's).
    """
    p = cl.prob_classify(getDict(text))
    return p.prob('C')


#read in the input tsv
try:
    data = read_table(sys.argv[1])
except OSError:
    print("The given input does not point to a valid file. Please try again! Input: "+sys.argv[1])
    print("Quitting.")
    sys.exit()
except IndexError:
    print("No input was passed; quitting. Come back with data for me to eat!")
    sys.exit()

#read in the classifier
file_path = 'equal-word-features-bayes.obj'
n_bytes = 2**31
max_bytes = 2**31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
cl = pickle.loads(bytes_in)

#calculate probabilities
texts = list(data.message)
probs = []
for t in texts:
    probs.append(classify(t))

#append probabilities to input tsv
data['probabilityClinton'] = probs
data.to_csv(sys.argv[1], index=False, sep="\t", doublequote=False, escapechar="\\")
