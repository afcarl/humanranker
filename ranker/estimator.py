from math import exp
from math import log
from sys import argv
import argparse
import numpy as np

# Regularization Parameters
item_mean = 0.0
item_std = 1.0
discrim_mean = 1.0
discrim_std = 1.0
bias_mean = 4.0
bias_std = 1.5
prec_mean = 1.0
prec_std = 1.0

class FakeObject:
    """
    A class for simulating the objects returned by Django querysets.
    """
    pass

def expz(val):
    """
    This function computes the exp of a value with the value bounded to
    (-12,12). This is useful in logistic models where values greater than these
    bounds have minimial effect on the predictive probabilities, but introduce
    overflow/underflow errors. 
    
    >>> expz(12)
    162754.79141900392

    >>> expz(13)
    162754.79141900392
    """
    if val > 12:
        return exp(12)
    if val < -12:
        return exp(-12)
    return exp(val)

def ll_combined(x, item_ids, judge_ids, pairwise=[], individual=[]):
    """
    This function computes the _negative_ loglikelihood of both the individual
    and pairwise data. 

    Keyword arguments:
    x -- the current parameter estimates.
    item_ids -- the ids of the items being evaluated
    judge_ids -- the ids of the judges being evaluted
    pairwise -- an iterator for the pairwise ratings
    individual -- an iterator for the individual ratings

    Without any ratings the model will return a likelihood based solely on the
    regularization parameters. 
    >>> ll_combined([0,0,1,1,3,1], [0,1], [0], [], [])
    4.0
    """
    item_val = {i:idx for idx, i in enumerate(item_ids)}
    discrim = {i:idx + len(item_val) for idx, i in enumerate(judge_ids)}
    bias = {i:idx + len(item_val) + len(judge_ids) for idx, i in enumerate(judge_ids)}
    precision = {i:idx + len(item_val) + 2*len(judge_ids) for idx, i in enumerate(judge_ids)}

    ll = 0.0
    for r in pairwise:
        left = x[item_val[r.left.id]]
        right = x[item_val[r.right.id]]
        d = x[discrim[r.judge.id]]

        y = r.value
        z = d * (left - right)
        p = 1 / (1 + expz(-1 * z))
        ll += y * log(p) + (1 - y) * log(1 - p)

    for l in individual:
        u = x[item_val[l.item.id]]
        b = x[bias[l.judge.id]]
        p = x[precision[l.judge.id]]

        ll += (1/2)*log(p) - (p * ((l.value - b - u) * (l.value - b - u)) / 2)

    # Regularization
    # Normal prior on means
    item_reg = 0.0
    for i in item_val:
        diff = x[item_val[i]] - item_mean
        item_reg += diff * diff
    item_reg = (-1.0 / (2 * item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = 0.0
    for i in discrim:
        diff = x[discrim[i]] - discrim_mean
        judge_reg += diff * diff
    judge_reg = ((-1.0 / (2 * discrim_std * discrim_std))
                 * judge_reg)

    # Normal prior on bias
    bias_reg = 0.0
    for i in bias:
        diff = x[bias[i]] - bias_mean
        bias_reg += diff * diff
    bias_reg = ((-1.0 / (2 * bias_std * bias_std))
                          * bias_reg)

    # Normal prior on precision
    prec_reg = 0.0
    for i in precision:
        diff = x[precision[i]] - prec_mean
        prec_reg += diff * diff
    prec_reg = ((-1.0 / (2 * prec_std * prec_std))
                            * prec_reg)

    return -1.0 * (ll + item_reg + judge_reg + bias_reg + prec_reg)

def ll_combined_grad(x, item_ids, judge_ids, pairwise=[], individual=[]):
    """
    This function computes the _negative_ gradient of the loglikelihood for
    each parameter in x, for both the individual and pairwise data. 
    
    Keyword arguments:
    x -- the current parameter estimates.
    item_ids -- the ids of the items being evaluated
    judge_ids -- the ids of the judges being evaluted
    pairwise -- an iterator for the pairwise ratings
    individual -- an iterator for the individual ratings

    >>> ll_combined_grad([0,0,1,1,3,1], [0,1], [0], [], [])
    array([-0.        , -0.        , -0.        , -1.33333333,  2.        , -0.        ])
    """
    item_val = {i:idx for idx, i in enumerate(item_ids)}
    discrim = {i:idx + len(item_val) for idx, i in enumerate(judge_ids)}
    bias = {i:idx + len(item_val) + len(judge_ids) for idx, i in enumerate(judge_ids)}
    precision = {i:idx + len(item_val) + 2*len(judge_ids) for idx, i in enumerate(judge_ids)}

    grad = np.array([0.0 for v in x])
    for r in pairwise:
        left = x[item_val[r.left.id]]
        right = x[item_val[r.right.id]]
        d = x[discrim[r.judge.id]]
        y = r.value
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))

        g = y - p 
        grad[item_val[r.left.id]] += d * g
        grad[item_val[r.right.id]] += -1 * d * g
        grad[discrim[r.judge.id]] += (left - right) * g

    for l in individual:
        u = x[item_val[l.item.id]]
        b = x[bias[l.judge.id]]
        prec = x[precision[l.judge.id]]
        #n = sqrt(1/prec)

        error = (l.value - b - u)
        grad[item_val[l.item.id]] += prec * error
        grad[bias[l.judge.id]] += prec * error
        grad[precision[l.judge.id]] += (1 / (2 * prec)) - (error * error)/2

    # Regularization
    # Normal prior on means
    item_reg = np.array([0.0 for v in x])
    for i in item_val:
        item_reg[item_val[i]] += (x[item_val[i]] - item_mean)
    item_reg = (-1.0 / (item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = np.array([0.0 for v in x])
    for i in discrim:
        judge_reg[discrim[i]] += (x[discrim[i]] - discrim_mean)
    judge_reg = (-1.0 / (discrim_std * discrim_std)) * judge_reg

    # Normal prior on bias
    bias_reg = np.array([0.0 for v in x])
    for i in bias:
        bias_reg[bias[i]] += (x[bias[i]] - bias_mean)
    bias_reg = (-1.0 / (bias_std * bias_std)) * bias_reg

    # Normal prior on noise
    prec_reg = np.array([0.0 for v in x])
    for i in precision:
        prec_reg[precision[i]] += (x[precision[i]] - prec_mean)
    prec_reg = (-1.0 / (prec_std * prec_std)) * prec_reg

    return -1 * (grad + item_reg + judge_reg + bias_reg + prec_reg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Estimate item parameters from individual and pairwise ratings.')
    parser.add_argument('-i', nargs=1, metavar='<individual-ratings.csv>',
                        required=False, type=argparse.FileType('r'), help='the individual ratings file')
    parser.add_argument('-p', nargs=1, metavar='<pairwise-ratings.csv>',
                        required=False, type=argparse.FileType('r'), help='the pairwise ratings file')

    parser.add_argument('-o', nargs=1, metavar='<output filename>',
                        required=False, type=argparse.FileType('w'), default="item_estimates.csv",
                        help='The filename where the estimates should be saved (default="item_estimates.csv").') 
    
    argsdata = parser.parse_args()

    print(argsdata.i)
    print(argsdata.p)
    print(argsdata.o)
    

