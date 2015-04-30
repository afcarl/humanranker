from math import exp
from math import log
import argparse
import numpy as np
from scipy.optimize import fmin_tnc

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
    
    individual = []
    pairwise = []
    judges = {}
    items = {}

    if argsdata.i:
        key = {}
        for row in argsdata.i[0]:
            data = row.rstrip().split(",")
            if len(key) == 0:
                for i,v in enumerate(data):
                    key[v] = i
                continue

            if int(data[key['item_id']]) in items:
                item = items[int(data[key['item_id']])]
            else:
                item = FakeObject()
                item.id = int(data[key['item_id']])
                item.name = data[key['item_name']]
                items[int(data[key['item_id']])] = item

            if int(data[key['judge_id']]) in judges:
                judge = judges[int(data[key['judge_id']])]
            else:
                judge = FakeObject()
                judge.id = int(data[key['judge_id']])
                judges[judge.id] = judge

            new_ind = FakeObject()            
            new_ind.item = item 
            new_ind.judge = judge
            new_ind.value = float(data[key['rating_value']])
            individual.append(new_ind)

    if argsdata.p:
        key = {}
        for row in argsdata.p[0]:
            data = row.rstrip().split(",")
            if len(key) == 0:
                for i,v in enumerate(data):
                    key[v] = i
                continue

            if int(data[key['left_item_id']]) in items:
                left_item = items[int(data[key['left_item_id']])]
            else:
                left_item = FakeObject()
                left_item.id = int(data[key['left_item_id']])
                left_item.name = data[key['left_item_name']] 
                items[int(data[key['left_item_id']])] = left_item

            if int(data[key['right_item_id']]) in items:
                right_item = items[int(data[key['right_item_id']])]
            else:
                right_item = FakeObject()
                right_item.id = int(data[key['right_item_id']])
                right_item.name = data[key['right_item_name']] 
                items[int(data[key['right_item_id']])] = right_item

            if int(data[key['judge_id']]) in judges:
                judge = judges[int(data[key['judge_id']])]
            else:
                judge = FakeObject()
                judge.id = int(data[key['judge_id']])
                judges[judge.id] = judge

            new_pair = FakeObject()            
            new_pair.left = left_item 
            new_pair.right= right_item 
            new_pair.judge = judge
            new_pair.value = float(data[key['rating_value']])
            pairwise.append(new_pair)

    # Estimate parameter values
    x0 = [item_mean for item in items] 
    x0 += [discrim_mean for j in judges]
    x0 += [bias_mean for j in judges]
    x0 += [prec_mean for j in judges]
    bounds = [('-inf','inf') for i in items]
    bounds += [(0.001,'inf') for j in judges]
    bounds += [('-inf','inf') for j in judges]
    bounds += [(0.001,'inf') for j in judges]

    iids = tuple(items)
    jids = tuple(judges)

    result = fmin_tnc(ll_combined, x0, 
                      fprime=ll_combined_grad, 
                      args=(iids, jids, pairwise, individual),
                      bounds=bounds,
                      disp=False)[0]

    item_val = {i: idx for idx, i in enumerate(iids)}
    discrim = {i: idx + len(iids) for idx, i in enumerate(jids)}
    bias = {i: idx + len(iids) + len(judges) for idx, i in enumerate(jids)}
    precision = {i: idx + len(iids) + 2 * len(judges) for idx, i in
                 enumerate(jids)}

    for i in items:
        items[i].mean = result[item_val[i]]
        items[i].conf = 10000.0

    for j in judges:
        judges[j].discrimination = result[discrim[j]]
        judges[j].bias = result[bias[j]]
        judges[j].precision = result[precision[j]]

    # compute the stds
    d2ll = np.array([0.0 for item in items])

    for r in pairwise:
        d = r.judge.discrimination
        left = r.left.mean
        right = r.right.mean
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))
        q = 1 - p
        d2ll[item_val[r.left.id]] += d * d * p * q
        d2ll[item_val[r.right.id]] += d * d * p * q
    
    for l in individual:
        d2ll[item_val[l.item.id]] += l.judge.precision

    # regularization terms
    for i,v in enumerate(d2ll):
        d2ll[i] += len(items) / (item_std * item_std) 

        #these shouldn't get included because they don't impact d2ll[i].
        #d2ll[i] += (len(judges) / (discrim_std * discrim_std))
        #d2ll[i] += (len(judges) / (bias_std * bias_std))
        #d2ll[i] += (len(judges) / (prec_std * prec_std))
    #print(d2ll)

    std = 1.0 / np.sqrt(d2ll)
    #print(std)

    for i in items:
        items[i].conf = 1.96 * std[item_val[i]]

    argsdata.o.write(",".join(['id', 'name', 'parameter estimate', 
                     '+/- 95% confidence interval']) + "\n")

    for i in items:
        argsdata.o.write(",".join([str(i), items[i].name, str(items[i].mean),
                                   str(items[i].conf)]) + "\n")

    

