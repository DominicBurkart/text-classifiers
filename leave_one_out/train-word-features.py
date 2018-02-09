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

import plotly
import plotly.graph_objs as go
from scipy.stats import kstest, ks_2samp, ttest_ind, pearsonr

verbose = False

def getFeatureSet(twoples):
    """
    @param twoples: the text / categorization from which the word featureset should be extracted.
    The format should look like [("Text with words", "categorization of text"), ("another text", "same/another category")]
    """
    global verbose
    out = [] #list of dictionary/label tuples
    for t in twoples:
        d = dict()
        text = t[0]
        if type(text) != str:
            if verbose:
                print("Non-string passed. Value: "+str(text))
                print("Ignoring value.")
        else:
            text = re.sub(r'\\x[A-Fa-f0-9]+', ' ', text) #removes characters transcribed from unicode to ascii (e.g. emoji)
            text = re.sub("[^a-zA-Z\d\s']", ' ', text) #only keeps alphanumeric characters
            words = text.lower().split()
            for w in words:
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
            out.append((d, t[1])) #appends a tuple with the dictionary object and the category to the out list
    return out #we have our feature set! send it out.

def classify_by(iv, dv, outname="classifier.obj", train_ratio = 0.8, show_feat = False):
    full = [(x, y) for x, y in zip(list(iv), list(dv))]
    random.shuffle(full)  # shuffles all documents to avoid order effects
    spliti = int(train_ratio * len(full))
    train = getFeatureSet(full[:spliti])
    test = getFeatureSet(full[spliti:])

    cl = nltk.NaiveBayesClassifier.train(train)
    print("Classifier accuracy: " + str(nltk.classify.accuracy(cl, test)))
    if show_feat:
        cl.show_most_informative_features(n=100)

    n_bytes = 2 ** 31  # the pickle package is slightly broken in python 3.4 and requires these params for saving large files.
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(cl)
    with open(outname, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

    return nltk.classify.accuracy(cl, test)


temps = []
spatials = []
socials = []
hypotheticals = []
controls = []
fr = read_csv("imagination_exp2_text.csv")

random.seed(5)
for s in range(100):
    temps.append(classify_by(fr.temporal, fr.group, 'exp2_temporal.obj'))
    spatials.append(classify_by(fr.spatial, fr.group, 'exp2_spatial.obj'))
    socials.append(classify_by(fr.social, fr.group, 'exp2_social.obj'))
    hypotheticals.append(classify_by(fr.hypothetical, fr.group, 'exp2_hypothetical.obj'))
    controls.append(classify_by(fr.control, fr.group, 'exp2_hypothetical.obj'))

fig = go.Figure(data=[go.Histogram(x=temps, histnorm='probability', name="Temporal"),
                      go.Histogram(x=spatials, histnorm='probability', name="Spatial"),
                      go.Histogram(x=socials, histnorm='probability', name="Social"),
                      go.Histogram(x=hypotheticals, histnorm='probability', name="Hypothetical"),
                      go.Histogram(x=controls, histnorm='probability', name="Control")],
                    layout = go.Layout(barmode='stack'))
plotly.offline.plot(fig, filename = "cross_validation.html")