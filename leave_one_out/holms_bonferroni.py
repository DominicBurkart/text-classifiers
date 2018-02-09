def multiple_comparison_dict(pvals, alpha=0.05):
    '''
    :param pvals: a list of p values to test
    :param alpha: the acceptable likelihood of a single type 1 error
    :return: a dictionary of the following lists: the significance (boolean), corresponding null p value,
    and input p value (in the same order as they were input).
    '''
    import numpy as np

    s = sorted(pvals)
    oo = list(np.argsort(pvals))

    truths = []
    correcteds = []
    reached = False
    for n in range(len(pvals)):
        c = alpha / (len(pvals) - n + 1)
        correcteds.append(c)
        if c > s[n] and not reached:
            truths.append(True)
        else:
            truths.append(False)
            reached = True

    cs = "corrected null p value with an alpha of " + str(alpha) + " and " + str(len(pvals)) + " comparisons"
    in_original_order = {"significant": [], "p value": [], cs: []}
    for i in range(len(oo)):
        in_original_order["significant"].append(truths[oo.index(i)])
        in_original_order["p value"].append(pvals[i])
        in_original_order[cs].append(correcteds[oo.index(i)])
    return in_original_order


def corrected_ps(n, alpha=0.05):
    '''
    returns the corrected p values for n comparisons
    :param n: integer
    :param alpha: acceptable total likelihood of a type I error
    :return: sorted list of null p values
    '''
    out = []
    for i in range(n):
        out.append(alpha / (n - i + 1))
    return out
