import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import fmin_tnc

from estimator import ll_combined
from estimator import ll_combined_grad
from estimator import expz
from estimator import item_std
from estimator import discrim_std
from estimator import bias_std
from estimator import prec_std
from estimator import FakeObject

def random_choice(judge, left_item, right_item):
    return random.choice([0,1]) 

def perfect_choice(judge, left_item, right_item):
    if left_item.value > right_item.value:
        return 1.0
    else:
        return 0.0

def noisy_choice(judge, left_item, right_item):
    prob_correct = 1.0 / (1.0 + expz(-judge.value * (left_item.value - right_item.value)))
    r = random.random()
    if r <= prob_correct:
        return 1.0
    else:
        return 0.0

def random_judgement(judge, item):
    return random.choice([1,2,3,4,5,6,7])

def perfect_judgement(judge, item):
    return 3.0

def noisy_judgement(judge, item):
    return 3.0

def random_pair(items):
    items = [i for i in items]
    random.shuffle(items)
    return items[0:2]

def conf_rate(items):
    items = [i for i in items]
    items.sort(key=lambda x: x.mean)

    item1 = None
    item2 = None
    diff = float('-inf')
    i1 = None
    i2 = None
    for item in items:
        i1 = i2
        i2 = item

        if not i1 or not i2:
            continue

        lower = max(i1.mean - i1.conf, i2.mean - i2.conf)
        upper = min(i1.mean + i1.conf, i2.mean + i2.conf)
        if upper - lower > diff:
            item1 = i1
            item2 = i2
            diff = upper - lower

    if random.random() > 0.5:
        temp = item1
        item1 = item2
        item2 = temp

    return item1, item2

def simulate(num_runs, num_items, num_judges, num_ratings, decision_fn, rate_fn):

    accuracy = [[] for i in range(num_ratings)]

    for run in range(num_runs):
        items = []
        ids = []
        for i in range(num_items):
            item = FakeObject()
            item.id = i
            item.value = random.normalvariate(0,1)
            #item.value = random.uniform(-100,100)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = FakeObject()
            judge.id = i
            judge.value = max(0.001, random.normalvariate(1,1))
            #judge.value = 7 
            #max(0.001,random.uniform(0,1))
            judge.cache = {}
            judge.decision_fn = decision_fn
            judges.append(judge)
            jids.append(i)

        # generate ratings
        ratings = []
        run_accuracy = []
        for i in range(num_ratings):

            x0 = [0.0 for item in items] 
            x0 += [1.0 for judge in judges]
            x0 += [0.0 for judge in judges]
            x0 += [0.0 for judge in judges]
            bounds = [('-inf','inf') for v in ids] 
            bounds += [(0.001,'inf') for v in jids]
            bounds += [('-inf','inf') for v in jids]
            bounds += [(0.001,'inf') for v in jids]
            
            result = fmin_tnc(ll_combined, x0, 
                              #approx_grad=True,
                              fprime=ll_combined_grad, 
                               args=(tuple(ids), tuple(jids), ratings, []),
                              bounds=bounds, disp=False)[0]

            
            for item in items:
                item.mean = result[ids[item.id]]
                item.conf = 10000.0

            if len(result) > len(items):
                for judge in judges:
                    judge.discrimination = result[jids[judge.id]]
            else:
                for judge in judges:
                    judge.discrimination = 1.0

            d2ll = np.array([0.0 for item in items])

            for r in ratings:
                d = r.judge.discrimination
                left = r.left.mean
                right = r.right.mean
                p = 1.0 / (1.0 + expz(-1 * d * (left-right)))
                q = 1 - p
                d2ll[ids[r.left.id]] += d * d * p * q
                d2ll[ids[r.right.id]] += d * d * p * q

            # regularization terms
            for i,v in enumerate(d2ll):
                d2ll[i] += len(ids) / (item_std * item_std) 
                
                if len(result) > len(items):
                    d2ll[i] += len(jids) / (discrim_std * discrim_std)
            #print(d2ll)

            std = 1.0 / np.sqrt(d2ll)
            #print(std)

            for item in items:
                item.conf = 1.96 * std[ids[item.id]]

            actual = np.array([item.value for item in items])
            predicted = np.array([item.mean for item in items])

            r = spearmanr(actual,predicted)[0]
            if math.isnan(r):
                run_accuracy.append(0.0)
            else:
                run_accuracy.append(r)

            r = FakeObject()
            r.left, r.right = rate_fn(items)
            r.judge = random.choice(judges)
            r.value = r.judge.decision_fn(r.judge, r.left, r.right)

            #fz = frozenset([r.left, r.right])
            #if fz in r.judge.cache:
            #    if r.judge.cache[fz] == r.left:
            #        r.value = 1
            #    else:
            #        r.value = 0
            #else:
            #    r.value = r.judge.decision_fn(r.judge, r.left, r.right)
            #    if r.value == 1:
            #        r.judge.cache[fz] = r.left
            #    else:
            #        r.judge.cache[fz] = r.right
            ratings.append(r)

        for idx, v in enumerate(run_accuracy):
            accuracy[idx].append(v)
        #reliability += np.array(run_reliability)

    #reliability /= num_runs

    return accuracy


if __name__ == "__main__":

    num_runs = 5 
    num_items = 40
    num_judges = 5 
    num_ratings = 200 

    # RANDOM
    reliability_random = simulate(num_runs, num_items, num_judges, num_ratings,
                           noisy_choice, random_pair)
    rel_random_mean = [np.mean(np.array(l)) for l in reliability_random]
    rel_random_lower = [rel_random_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_random)]
    rel_random_upper = [rel_random_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_random)]
    plt.fill_between([i for i in range(num_ratings)], rel_random_lower,
                     rel_random_upper, alpha=0.5, facecolor="blue")
    plt.plot([i for i in range(num_ratings)], rel_random_mean,
             label="Pairwise Rating", color="blue")

    plt.title("Simulated Accuracy for " + str(num_items) + " Items and " +
              str(num_judges) + " Judges (Avg of " + str(num_runs) + " Runs)")
    plt.xlabel("# of ratings")
    plt.ylabel("Spearman's Rank Correlation Coefficient")
    plt.legend(loc=4)
    plt.show()


