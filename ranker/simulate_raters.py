import sys, os
sys.path.append("/Users/cmaclell/Projects/pairwise/pairwise")
sys.path.append("/Users/cmaclell/Projects/pairwise/")
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'

from views import ll_2p, ll_2p_grad, expz, item_std, judge_std
from scipy.optimize import fmin_tnc
import random
import math
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

class Foo:
    pass

def random_evaluate(judge, left_item, right_item):
    return random.choice([0,1]) 

def perfect_evaluate(judge, left_item, right_item):
    if left_item.value > right_item.value:
        return 1.0
    else:
        return 0.0

def noisy_evaluate(judge, left_item, right_item):
    prob_correct = 1.0 / (1.0 + expz(-judge.value * (left_item.value - right_item.value)))
    r = random.random()
    #print("Probability of correct:", prob_correct, r)
    if r <= prob_correct:
        return 1.0
    else:
        return 0.0

def random_rate(items):
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

    reliability = [[] for i in range(num_ratings)]
    #reliability = np.array([0.0 for i in range(num_runs)])

    for run in range(num_runs):
        items = []
        ids = []
        for i in range(num_items):
            item = Foo()
            item.id = i
            item.value = random.normalvariate(0,1)
            #item.value = random.uniform(-100,100)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = Foo()
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
        run_reliability = []
        for i in range(num_ratings):

            x0 = [0.0 for item in items] + [1.0 for judge in judges]
            bounds = [('-inf','inf') for v in ids] + [(0.001,'inf') for v in jids]
            result = fmin_tnc(ll_2p, x0, 
                              #approx_grad=True,
                              fprime=ll_2p_grad, 
                              args=(tuple(jids), tuple(ids), ratings), bounds=bounds,
                              disp=False)[0]

            
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
                    d2ll[i] += len(jids) / (judge_std * judge_std)
            #print(d2ll)

            std = 1.0 / np.sqrt(d2ll)
            #print(std)

            for item in items:
                item.conf = 1.96 * std[ids[item.id]]

            actual = np.array([item.value for item in items])
            predicted = np.array([item.mean for item in items])

            r = spearmanr(actual,predicted)[0]
            if math.isnan(r):
                run_reliability.append(0.0)
            else:
                run_reliability.append(r)

            r = Foo()
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

        for idx, v in enumerate(run_reliability):
            reliability[idx].append(v)
        #reliability += np.array(run_reliability)

    #reliability /= num_runs

    return reliability


if __name__ == "__main__":

    num_runs = 20 
    num_items = 40
    num_judges = 5 
    num_ratings = 50 

    # CONF INIT
    reliability_conf = simulate(num_runs, num_items, num_judges, num_ratings,
                           noisy_evaluate, conf_rate)

    rel_conf_mean = [np.mean(np.array(l)) for l in reliability_conf]
    rel_conf_lower = [rel_conf_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_conf)]
    rel_conf_upper = [rel_conf_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_conf)]
    plt.fill_between([i for i in range(num_ratings)], rel_conf_lower,
                     rel_conf_upper, alpha=0.5, facecolor="green")
    plt.plot([i for i in range(num_ratings)], rel_conf_mean,
             label="Overlapping Confidence Pairs", color="green")

    # RANDOM
    reliability_random = simulate(num_runs, num_items, num_judges, num_ratings,
                           noisy_evaluate, random_rate)
    rel_random_mean = [np.mean(np.array(l)) for l in reliability_random]
    rel_random_lower = [rel_random_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_random)]
    rel_random_upper = [rel_random_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(reliability_random)]
    plt.fill_between([i for i in range(num_ratings)], rel_random_lower,
                     rel_random_upper, alpha=0.5, facecolor="blue")
    plt.plot([i for i in range(num_ratings)], rel_random_mean,
             label="Random Pairs", color="blue")

    plt.title("Simulated Accuracy for " + str(num_items) + " Items and " +
              str(num_judges) + " Judges (Avg of " + str(num_runs) + " Runs)")
    plt.xlabel("# of pairwise comparisons")
    plt.ylabel("Spearman's Rank Correlation Coefficient")
    plt.legend(loc=4)
    plt.show()


