"""
Naive Bayes Classifier that considers each alphanumeric word as a distinct feature,
with no special treatment to specific word locations (e.g. first word) etc.

Removes emoji text that has been transcribed into unicode (e.g. \x99).

Can perform leave-one-out cross-validation or randomly split the data into a training and test set according to a given
ratio.

Can print the accuracy of the classifer and 100 most informative features to the console.

Can save the trained classifier in the working directory.

Dominic Burkart dominicburkart@nyu.edu
"""

import pickle
import random
import re

import nltk
from pandas import *

_input_file_ = "imagination_exp2_text.csv"

local = True
verbose = False


def getFeatureSet(twoples):
    """
    @param twoples: the text / categorization from which the word featureset should be extracted.
    The format should look like [("Text with words", "categorization of text"), ("another text", "same/another category")]
    """
    global verbose
    out = []  # list of dictionary/label tuples
    for t in twoples:
        d = dict()
        text = t[0]
        if type(text) != str:
            if verbose:
                print("Non-string passed. Value: " + str(text))
                print("Ignoring value.")
        else:
            text = re.sub(r'\\x[A-Fa-f0-9]+', ' ',
                          text)  # removes characters transcribed from unicode to ascii (e.g. emoji)
            text = re.sub("[^a-zA-Z\d\s']", ' ', text)  # only keeps alphanumeric characters
            words = text.lower().split()
            for w in words:
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
            out.append((d, t[1]))  # appends a tuple with the dictionary object and the category to the out list
    return out  # we have our feature set! send it out.


def classify_by(iv, dv, outname="classifier.obj", train_ratio=0.8, show_feat=False, save_classifier=True):
    '''
    for splitting the data by test and training sets randomly and giving classification accuracy on the test set.

    :param iv:
    :param dv:
    :param outname:
    :param train_ratio:
    :param show_feat:
    :param save_classifier:
    :return:
    '''
    full = [(x, y) for x, y in zip(list(iv), list(dv))]
    random.shuffle(full)  # shuffles all documents to avoid order effects
    spliti = int(train_ratio * len(full))
    assert spliti < len(full) - 1
    assert spliti > 1
    train = getFeatureSet(full[:spliti])
    test = getFeatureSet(full[spliti:])

    cl = nltk.NaiveBayesClassifier.train(train)
    print("Classifier accuracy: " + str(nltk.classify.accuracy(cl, test)))
    if show_feat:
        cl.show_most_informative_features(n=100)

    if save_classifier:
        n_bytes = 2 ** 31  # the pickle package is slightly broken in python 3.4 and requires these params for saving large files.
        max_bytes = 2 ** 31 - 1
        bytes_out = pickle.dumps(cl)
        with open(outname, 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])

    return nltk.classify.accuracy(cl, test)


def hypt(accuracy, iv, dv, perms=10000, show_graph=True, name="hyptest", print_progress=True,
         multiprocess=True, save_perm_accuracies=True):
    '''
    Tests whether classifiers are performing significantly better than chance.

    Permutation-based hypothesis testing based on removing the correspondence between the IV and the DV via
    randomization to generate a null distribution.

    :param accuracy:
    :param iv:
    :param dv:
    :param perms:
    :param show_graph:
    :param plot_name:
    :param print_progress:
    :return:
    '''
    import copy
    null_accuracy = []
    if multiprocess:
        import multiprocessing
        async_kwargs = {"hyp_test": False,
                        "show_graph": False,
                        "write_out": False}
        print("Instantiating multiprocessing for " + str(perms) + " permutations on " +
              str(multiprocessing.cpu_count()) + " cores.")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            out_dicts = []
            resps = []
            for i in range(perms):
                civ = copy.deepcopy(iv)
                np.random.shuffle(civ)
                resps.append(
                    pool.apply_async(leave_one_out,
                                     args=(civ, dv),
                                     kwds=async_kwargs,
                                     callback=out_dicts.append))
            for r in resps:
                r.wait()
            null_accuracy = [d['accuracy'] for d in out_dicts]
    else:
        for i in range(perms):
            civ = copy.deepcopy(iv)
            np.random.shuffle(civ)
            null_accuracy.append(leave_one_out(civ, dv, show_graph=False, hyp_test=False, write_out=False)['accuracy'])
            if print_progress:
                print("Permutation test iteration #: " + str(i + 1))
                print("Percent complete: " + str(((i + 1) / perms) * 100) + "%")
    if show_graph:
        # todo: add line to show where the classifier's average accuracy was
        import plotly.graph_objs as go
        from plotly.offline import plot
        fig = go.Figure(data=[go.Histogram(x=null_accuracy, opacity=0.9)])
        plot(fig, filename=name + ".html")
    g = [s for s in null_accuracy if s >= accuracy]
    if save_perm_accuracies:
        import csv
        with open(name + '_null_accuracies.csv', "w") as f:
            w = csv.writer(f)
            for a in null_accuracy:
                w.writerow([a])
    return len(g) / len(null_accuracy)  # probability of null hypothesis


def leave_one_out(iv, dv, name=None, show_feat=False, hyp_test=True, show_graph=False, write_out=True):
    '''
    Leave one out cross-validated classification.

    :param iv:
    :param dv:
    :param name:
    :param show_feat:
    :param hyp_test:
    :param show_graph:
    :param write_out:
    :return:
    '''
    global verbose
    full = [(x, y) for x, y in zip(list(iv), list(dv))]
    v = []
    for iout in range(len(full)):
        featset = getFeatureSet([full[i] for i in range(len(full)) if i != iout])
        test = getFeatureSet([full[iout]])
        cl = nltk.NaiveBayesClassifier.train(featset)
        if len(test) != 0:
            result = 1 if cl.classify(test[0][0]) == test[0][1] else 0  # correct or false
            v.append(result)
            if show_feat:
                cl.show_most_informative_features(n=100)
        elif verbose:
            print("At least one testing value was empty. Ignoring that testing value.")

    a = np.mean(v)
    if hyp_test:
        if name is None:
            out = {"accuracy": a, "null probability": hypt(a, iv, dv, show_graph=show_graph)}
        else:
            out = {"accuracy": a,
                   "null probability": hypt(a, iv, dv, name=name, show_graph=show_graph),
                   "name": name}
    else:
        if name is None:
            out = {"accuracy": a}
        else:
            out = {"accuracy": a, "name": name}
    if write_out:
        if name is None:
            print("Name for this leave one out analysis is not defined. Data may be overwritten.")
            DataFrame([out]).to_csv("loo.csv")
        else:
            DataFrame([out]).to_csv(name + ".csv")
    return out


if __name__ == "__main__":
    fr = read_csv(_input_file_)

    if local:
        l = [
            leave_one_out(fr.spatial, fr.group, "spatial_loo"),
            leave_one_out(fr.social, fr.group, "social_loo"),
            leave_one_out(fr.hypothetical, fr.group, "hypothetical_loo"),
            leave_one_out(fr.control, fr.group, "control_loo"),
            leave_one_out(fr.temporal, fr.group, "temporal_loo")
        ]
        print(l)
        df = DataFrame(l)
        df.to_csv("classifier_performance.csv")
    else:
        import sbatch

        sbatch.load("leave_one_out", [fr.temporal, fr.group, "temporal_loo"])
        sbatch.load("leave_one_out", [fr.spatial, fr.group, "spatial_loo"])
        sbatch.load("leave_one_out", [fr.social, fr.group, "social_loo"])
        sbatch.load("leave_one_out", [fr.hypothetical, fr.group, "hypothetical_loo"])
        sbatch.load("leave_one_out", [fr.control, fr.group, "control_loo"])
        sbatch.launch()
