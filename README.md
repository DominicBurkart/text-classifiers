# text_classifiers
text classification programs


leave_one_out contains a variety of classification schemes designed to take in text input, calculate accuracy via a leave-one-out approach with a variety of bag-of-words classifiers, and then perform permutive significance testing by randomizing the label and performing the same leave-one-out approach 100,000 times (by removing the correspondence between the classification label and the features (words), this randomization of the label generates a null distribution against which the finding of the single leave-one-out operation with the actual label-feature correspondence can be compared). This folder also contains simple functions for performing multiple comparison corrections using the holm-bonferroni method, which were relevant for the specific, original application of these classifiers.
