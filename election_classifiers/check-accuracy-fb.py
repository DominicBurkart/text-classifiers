"""
Used for running the equal word features classifer (created by train-word-features.py).

USAGE:

1.) make sure that the current directory is the folder that contains this
file and the other bayes encoding files in this suite.

2.) enter the command "python3 check-accuracy-fb.py path/to/input.tsv",
where path/to/input.tsv is replaced with the path to the text to be encoded.
If the filepath includes spaces, you need to put it in quotes (path/to/input.tsv vs "path/to/in put.tsv").



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
    try:
        p = cl.prob_classify(getDict(text))
        return p.prob('C')
    except TypeError:
        print("One line of input could not be interpreted as text. Excluding.")
        return "NA"

is_tsv = True;

#read in the input tsv
try:
    if (sys.argv[1].lower().endswith("tsv")):
        data = read_table(sys.argv[1])
    elif (sys.argv[1].lower().endswith("csv")):
        is_tsv = False;
        data = read_csv(sys.argv[1])
    else:
        print("The input filetype could not be inferred by the file name. attempting to parse as TSV.")
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
texts = list(set(list((data.post))))
probs = []
for t in texts:
    probs.append(classify(t))

print("average probability clinton: "+str(sum(probs)/len(probs)))

above = 0
for p in probs:
    if p > 0.5:
        above = above + 1

print("number categorized as clinton: "+str(above/len(probs))) 

print(set(list(data.post_author))) #confirm that only Donald Trump wrote posts in data
