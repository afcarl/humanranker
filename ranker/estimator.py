from math import exp
from math import log
import numpy as np

# regularization parameters
item_mean = 0.0
item_std = 1.0

discrim_mean = 1.0
discrim_std = 1.0

bias_mean = 4.0
bias_std = 1.5

prec_mean = 1.0
prec_std = 1.0

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
    This function computes the loglikelihood of both the individual and pairwise
    data. 
    
    The loglikelihood of the pairwise rating is computed using the following
    equation:
        ll_pair = sum_i^n y[i] * log(p[i]) + (1 - y[i]) * log(1 - p[i])
    assuming the dependent value follows a binomial distribution.

    The loglikelihood of the individual ratings is computed using the following
    equation:
        ll_ind = sum_i^n ( (1/2) * log(std_{judge[i]}) - (std_{judge[i]} / 2) 
                            * (y[i] - bias_{judge[i]} - mean_{item[i]})^2 )
    assuming a linear model.

    This function returns the loglikelihood of all of the data as the sum of
    the two loglikelihood functions (i.e., ll_pair + ll_ind), where the item
    parameters are shared across both models. Additionally, the system adds a
    regularization term based on the global regularization values.

    Keyword arguments:
    x -- the current parameter estimates.
    item_ids -- the ids of the items being evaluated
    judge_ids -- the ids of the judges being evaluted
    pairwise -- an iterator for the pairwise ratings
    individual -- an iterator for the individual ratings

    >>> ll_combined([0,0,1,1,3,1], [0,1], [0], [], [])
    4.0
    """
    ids = {i:idx for idx, i in enumerate(item_ids)}
    discids = {i:idx + len(ids) for idx, i in enumerate(judge_ids)}
    biasids = {i:idx + len(ids) + len(discids) for idx, i in enumerate(judge_ids)}
    precids = {i:idx + len(ids) + 2*len(discids) for idx, i in enumerate(judge_ids)}
    ratings = pairwise
    likerts = individual

    ll = 0.0
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        d = x[discids[r.judge.id]]

        y = r.value
        z = d * (left - right)
        p = 1 / (1 + expz(-1 * z))
        ll += y * log(p) + (1 - y) * log(1 - p)

    for l in likerts:
        u = x[ids[l.item.id]]
        b = x[biasids[l.judge.id]]
        p = x[precids[l.judge.id]]
        #n = sqrt(1/p)

        ll += (1/2)*log(p) - (p * ((l.value - b - u) * (l.value - b - u)) / 2)

    # Regularization
    # Normal prior on means
    item_reg = 0.0
    for i in ids:
        diff = x[ids[i]] - item_mean
        item_reg += diff * diff
    item_reg = (-1.0 / (2 * item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = 0.0
    for i in discids:
        diff = x[discids[i]] - discrim_mean
        judge_reg += diff * diff
    judge_reg = ((-1.0 / (2 * discrim_std * discrim_std))
                 * judge_reg)

    # Normal prior on bias
    bias_reg = 0.0
    for i in biasids:
        diff = x[biasids[i]] - bias_mean
        bias_reg += diff * diff
    bias_reg = ((-1.0 / (2 * bias_std * bias_std))
                          * bias_reg)

    # Normal prior on precision
    prec_reg = 0.0
    for i in precids:
        diff = x[precids[i]] - prec_mean
        prec_reg += diff * diff
    prec_reg = ((-1.0 / (2 * prec_std * prec_std))
                            * prec_reg)

    return -1.0 * (ll + item_reg + judge_reg + bias_reg + prec_reg)

def ll_combined_grad(x, *args):
    ids = {i:idx for idx, i in enumerate(args[0])}
    discids = {i:idx + len(ids) for idx, i in enumerate(args[1])}
    biasids = {i:idx + len(ids) + len(discids) for idx, i in enumerate(args[1])}
    precids = {i:idx + len(ids) + 2*len(discids) for idx, i in enumerate(args[1])}
    ratings = args[2]
    likerts = args[3]
    #ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    grad = np.array([0.0 for v in x])
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        d = x[discids[r.judge.id]]
        y = r.value
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))

        g = y - p 
        grad[ids[r.left.id]] += d * g
        grad[ids[r.right.id]] += -1 * d * g
        grad[discids[r.judge.id]] += (left - right) * g

    for l in likerts:
        u = x[ids[l.item.id]]
        b = x[biasids[l.judge.id]]
        prec = x[precids[l.judge.id]]
        #n = sqrt(1/prec)

        error = (l.value - b - u)
        grad[ids[l.item.id]] += prec * error
        grad[biasids[l.judge.id]] += prec * error
        grad[precids[l.judge.id]] += (1 / (2 * prec)) - (error * error)/2

    # Regularization
    # Normal prior on means
    item_reg = np.array([0.0 for v in x])
    for i in ids:
        item_reg[ids[i]] += (x[ids[i]] - item_mean)
    item_reg = (-1.0 / (item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = np.array([0.0 for v in x])
    for i in discids:
        judge_reg[discids[i]] += (x[discids[i]] - discrim_mean)
    judge_reg = (-1.0 / (discrim_std * discrim_std)) * judge_reg

    # Normal prior on bias
    bias_reg = np.array([0.0 for v in x])
    for i in biasids:
        bias_reg[biasids[i]] += (x[biasids[i]] - bias_mean)
    bias_reg = (-1.0 / (bias_std * bias_std)) * bias_reg

    # Normal prior on noise
    prec_reg = np.array([0.0 for v in x])
    for i in precids:
        prec_reg[precids[i]] += (x[precids[i]] - prec_mean)
    prec_reg = (-1.0 / (prec_std * prec_std)) * prec_reg

    return -1 * (grad + item_reg + judge_reg + bias_reg + prec_reg)

def ll_2p(x, *args):
    ids = {i:idx for idx, i in enumerate(args[0])}
    jids = {i:idx + len(ids) for idx, i in enumerate(args[1])}
    ratings = args[2]
    #ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    ll = 0.0
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        d = x[jids[r.judge.id]]

        y = r.value
        z = d * (left - right)
        p = 1 / (1 + expz(-1 * z))
        ll += y * log(p) + (1 - y) * log(1 - p)

    # Regularization
    # Normal prior on means
    item_reg = 0.0
    for i in ids:
        diff = x[ids[i]] - item_mean
        item_reg += diff * diff
    item_reg = (-1.0 / (2 * item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = 0.0
    for i in jids:
        diff = x[jids[i]] - discrim_mean
        judge_reg += diff * diff
    judge_reg = ((-1.0 / (2 * discrim_std * discrim_std))
                 * judge_reg)

    return -1.0 * (ll + item_reg + judge_reg)

def ll_2p_grad(x, *args):
    ids = {i:idx for idx, i in enumerate(args[0])}
    jids = {i:idx + len(ids) for idx, i in enumerate(args[1])}
    ratings = args[2]
    #ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    grad = np.array([0.0 for v in x])
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        d = x[jids[r.judge.id]]
        y = r.value
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))

        g = y - p 
        grad[ids[r.left.id]] += d * g
        grad[ids[r.right.id]] += -1 * d * g
        grad[jids[r.judge.id]] += (left - right) * g

    # Regularization
    # Normal prior on means
    item_reg = np.array([0.0 for v in x])
    for i in ids:
        item_reg[ids[i]] += (x[ids[i]] - item_mean)
    item_reg = (-1.0 / (item_std * item_std)) * item_reg

    # Gamma prior on discriminations
    judge_reg = np.array([0.0 for v in x])
    for i in jids:
        judge_reg[jids[i]] += (x[jids[i]] - discrim_mean)
    judge_reg = (-1.0 / (discrim_std * discrim_std)) * judge_reg

    return -1 * (grad + item_reg + judge_reg)

