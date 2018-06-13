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
from sklearn.naive_bayes import MultinomialNB

_input_file_ = "~/Documents/princeton/imagination/exps_2_and_3_text.csv"

local = False
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


def hypt(accuracy, source_iv, source_dv, target_iv, target_dv, perms=10000, show_graph=True, name="decode",
         print_progress=True, multiprocess=True, save_perm_accuracies=True):
    '''
    Tests whether classifiers are performing significantly better than chance cross-domain.

    Permutation-based hypothesis testing based on removing the correspondence between the IV and the DV via
    randomization to generate a null distribution of performance for each target domain.

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
    source_iv = np.array(source_iv)
    source_dv = np.array(source_dv)
    target_iv = np.array(target_iv)
    target_dv = np.array(target_dv)
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
                reindex = np.array(range(len(source_iv)))
                np.random.shuffle(reindex)
                sciv = source_iv[reindex]
                tciv = target_iv[reindex]
                resps.append(
                    pool.apply_async(cross_decode,
                                     args=(sciv, source_dv, tciv, target_dv),
                                     kwds=async_kwargs,
                                     callback=out_dicts.append))
            for r in resps:
                r.wait()
            null_accuracy = [d['accuracy'] for d in out_dicts]
    else:
        for i in range(perms):
            reindex = np.array(np.random.shuffle(range(len(source_iv))))
            sciv = source_iv[reindex]
            tciv = target_iv[reindex]
            null_accuracy.append(cross_decode(sciv, source_dv,
                                              tciv, source_iv,
                                              show_graph=False,
                                              hyp_test=False,
                                              write_out=False)['accuracy'])
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


def cross_decode(source_iv, source_dv, target_iv, target_dv, name=None,
                 show_feat=False, hyp_test=True, show_graph=False, write_out=True):
    '''
    Leave one out cross-validated classification across dimensions. This means that we're testing in a different
    dimension and with a left-out participant.

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
    source_full = [(x, y) for x, y in zip(list(source_iv), list(source_dv))]
    target_full = [(x, y) for x, y in zip(list(target_iv), list(target_dv))]
    assert len(target_full) == len(source_full)

    v = []
    for iout in range(len(source_full)):
        featset = getFeatureSet([source_full[i] for i in range(len(source_full)) if i != iout])
        test = getFeatureSet([target_full[iout]])
        cl = nltk.classify.scikitlearn.SklearnClassifier(MultinomialNB()).train(featset)
        if len(test) != 0:
            result = 1 if cl.classify(test[0][0]) == test[0][1] else 0  # correct (1) or false (0)
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
                   "null probability": hypt(a, source_iv, source_dv, target_iv, target_dv,
                                            name=name, show_graph=show_graph),
                   "name": name}
    else:
        if name is None:
            out = {"accuracy": a}
        else:
            out = {"accuracy": a, "name": name}
    if write_out:
        if name is None:
            print("Name for this leave one out analysis is not defined. Data may be overwritten.")
            DataFrame([out]).to_csv("cross_decode.csv")
        else:
            DataFrame([out]).to_csv(name + ".csv")
    return out

if __name__ == "__main__":
    fr = read_csv(_input_file_)

    columns = [fr.spatial, fr.social, fr.hypothetical, fr.control, fr.temporal]
    flattened = []
    for s in columns:
        flattened.extend(s)
    flattened = DataFrame({"text": flattened, "group": list(fr.group) * len(columns)})

    dims = [
        fr.spatial,
        fr.social,
        fr.hypothetical,
        fr.control,
        fr.temporal
    ]

    names = [
        "spatial",
        "social",
        "hypothetical",
        "control",
        "temporal"
    ]

    if local:
        outs = []
        for i1 in range(len(dims)):
            for i2 in range(len(dims)):
                outs.append(cross_decode(dims[i1], fr.group, dims[i2], fr.group, names[i1] + "_to_" + names[i2]))
        df = DataFrame(outs)
        df.to_csv("cross_encoding_performance.csv")
    else:
        import sbatch
        for i1 in range(len(dims)):
            for i2 in range(len(dims)):
                sbatch.load("cross_decode", [dims[i1], fr.group, dims[i2], fr.group, names[i1] + "_to_" + names[i2]])
        sbatch.launch()
        print("jobs submitted to the cluster. Run the collation scripts when they're done!")

