import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import fmin_tnc

from estimator import ll_combined
from estimator import ll_combined_grad
from estimator import expz
from estimator import item_mean
from estimator import item_std
from estimator import discrim_mean
from estimator import discrim_std
from estimator import bias_mean
from estimator import bias_std
from estimator import prec_mean
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
    prob_correct = (1.0 / (1.0 + expz(-judge.true_discrim * (left_item.value -
                                                             right_item.value))))
    r = random.random()
    if r <= prob_correct:
        return 1.0
    else:
        return 0.0

def random_judgement(judge, item):
    return random.choice([1,2,3,4,5,6,7])

def perfect_judgement(judge, item):
    return max(1, min(7, round((item.value + bias_mean))))

def noisy_judgement(judge, item):
    estimate = random.normalvariate(item.value + judge.true_bias,
                                    math.sqrt(1/judge.true_precision))
    return max(1, min(7, round((estimate))))

def random_item(items):
    items = [i for i in items]
    random.shuffle(items)
    return items[0]

def random_pair(items):
    items = [i for i in items]
    random.shuffle(items)
    return items[0:2]

def simulate_individual(num_runs, num_items, num_judges, num_ratings,
                        estimation_mod, decision_fn, rate_fn):

    print("Simulating Individual")

    accuracy = [[] for i in range(num_ratings)]

    for run in range(num_runs):
        print("Run %i" % run)
        items = []
        ids = []
        for i in range(num_items):
            item = FakeObject()
            item.id = i
            item.value = random.normalvariate(item_mean,item_std)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = FakeObject()
            judge.id = i
            judge.true_bias = min(7,max(1,random.normalvariate(bias_mean, bias_std)))
            judge.true_precision = max(0.001, random.normalvariate(prec_mean,prec_std))
            judge.cache = {}
            judge.decision_fn = decision_fn
            judges.append(judge)
            jids.append(i)

        # generate ratings
        ratings = []
        run_accuracy = []
        result = None
        for i in range(num_ratings):

            if i % estimation_mod is 0:
                if result is None:
                    x0 = [0.0 for item in items] 
                    x0 += [1.0 for judge in judges]
                    x0 += [0.0 for judge in judges]
                    x0 += [1.0 for judge in judges]
                else:
                    x0 = result

                bounds = [('-inf','inf') for v in ids] 
                bounds += [(0.001,'inf') for v in jids]
                bounds += [('-inf','inf') for v in jids]
                bounds += [(0.001,'inf') for v in jids]
                
                result = fmin_tnc(ll_combined, x0, 
                                  #approx_grad=True,
                                  fprime=ll_combined_grad, 
                                  args=(tuple(ids), tuple(jids), [], ratings),
                                  bounds=bounds, disp=False)[0]

                ids = {i: idx for idx, i in enumerate(ids)}
                discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
                biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
                precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}
                
                for item in items:
                    item.mean = result[ids[item.id]]
                    item.conf = 10000.0

                for judge in judges:
                    judge.discrimination = result[discids[judge.id]]
                    judge.bias = result[biasids[judge.id]]
                    judge.precision = result[precids[judge.id]]

                d2ll = np.array([0.0 for item in items])

                for l in ratings:
                    d2ll[ids[l.item.id]] += l.judge.precision

                # regularization terms
                for i,v in enumerate(d2ll):
                    d2ll[i] += len(ids) / (item_std * item_std) 
                    
                std = 1.0 / np.sqrt(d2ll)

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
            r.item = rate_fn(items)
            r.judge = random.choice(judges)
            r.value = r.judge.decision_fn(r.judge, r.item)
            ratings.append(r)

        for idx, v in enumerate(run_accuracy):
            accuracy[idx].append(v)

    return accuracy

def simulate_pairwise(num_runs, num_items, num_judges, num_ratings,
                      estimation_mod, decision_fn, rate_fn):

    print("Simulating Pairwise")

    accuracy = [[] for i in range(num_ratings)]

    for run in range(num_runs):
        print("Run %i" % run)
        items = []
        ids = []
        for i in range(num_items):
            item = FakeObject()
            item.id = i
            item.value = random.normalvariate(item_mean,item_std)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = FakeObject()
            judge.id = i
            judge.true_discrim = max(0.001, random.normalvariate(discrim_mean,discrim_std))
            judge.cache = {}
            judge.decision_fn = decision_fn
            judges.append(judge)
            jids.append(i)

        # generate ratings
        ratings = []
        run_accuracy = []
        result = None
        for i in range(num_ratings):

            if i % estimation_mod is 0:
                if result is None:
                    x0 = [0.0 for item in items] 
                    x0 += [1.0 for judge in judges]
                    x0 += [0.0 for judge in judges]
                    x0 += [1.0 for judge in judges]
                else:
                    x0 = result

                bounds = [('-inf','inf') for v in ids] 
                bounds += [(0.001,'inf') for v in jids]
                bounds += [('-inf','inf') for v in jids]
                bounds += [(0.001,'inf') for v in jids]
                
                result = fmin_tnc(ll_combined, x0, 
                                  #approx_grad=True,
                                  fprime=ll_combined_grad, 
                                   args=(tuple(ids), tuple(jids), ratings, []),
                                  bounds=bounds, disp=False)[0]

                ids = {i: idx for idx, i in enumerate(ids)}
                discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
                biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
                precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}
                
                for item in items:
                    item.mean = result[ids[item.id]]
                    item.conf = 10000.0

                for judge in judges:
                    judge.discrimination = result[discids[judge.id]]
                    judge.bias = result[biasids[judge.id]]
                    judge.precision = result[precids[judge.id]]

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
                    
                std = 1.0 / np.sqrt(d2ll)

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

            ratings.append(r)

        for idx, v in enumerate(run_accuracy):
            accuracy[idx].append(v)

    return accuracy

def simulate_combined(num_runs, num_items, num_judges, num_ratings,
                      estimation_mod, 
                      pairwise_decision_fn, pairwise_rate_fn,
                      individual_decision_fn, individual_rate_fn):

    print("Simulating Combined")

    accuracy = [[] for i in range(num_ratings)]

    for run in range(num_runs):
        print("Run %i" % run)
        items = []
        ids = []
        for i in range(num_items):
            item = FakeObject()
            item.id = i
            item.value = random.normalvariate(item_mean,item_std)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = FakeObject()
            judge.id = i
            judge.true_discrim = max(0.001, random.normalvariate(discrim_mean,discrim_std))
            judge.true_bias = min(7,max(1,random.normalvariate(bias_mean, bias_std)))
            judge.true_precision = max(0.001, random.normalvariate(prec_mean,prec_std))
            judge.cache = {}
            judge.pairwise_decision_fn = pairwise_decision_fn
            judge.individual_decision_fn = individual_decision_fn
            judges.append(judge)
            jids.append(i)

        # generate ratings
        pairwise = []
        individual = []
        run_accuracy = []
        result = None
        for i in range(num_ratings):

            if i % estimation_mod is 0:
                if result is None:
                    x0 = [0.0 for item in items] 
                    x0 += [1.0 for judge in judges]
                    x0 += [0.0 for judge in judges]
                    x0 += [1.0 for judge in judges]
                else:
                    x0 = result

                bounds = [('-inf','inf') for v in ids] 
                bounds += [(0.001,'inf') for v in jids]
                bounds += [('-inf','inf') for v in jids]
                bounds += [(0.001,'inf') for v in jids]
                
                result = fmin_tnc(ll_combined, x0, 
                                  #approx_grad=True,
                                  fprime=ll_combined_grad, 
                                   args=(tuple(ids), tuple(jids), pairwise,
                                         individual),
                                  bounds=bounds, disp=False)[0]

                ids = {i: idx for idx, i in enumerate(ids)}
                discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
                biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
                precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}
                
                for item in items:
                    item.mean = result[ids[item.id]]
                    item.conf = 10000.0

                for judge in judges:
                    judge.discrimination = result[discids[judge.id]]
                    judge.bias = result[biasids[judge.id]]
                    judge.precision = result[precids[judge.id]]

                d2ll = np.array([0.0 for item in items])

                for r in pairwise:
                    d = r.judge.discrimination
                    left = r.left.mean
                    right = r.right.mean
                    p = 1.0 / (1.0 + expz(-1 * d * (left-right)))
                    q = 1 - p
                    d2ll[ids[r.left.id]] += d * d * p * q
                    d2ll[ids[r.right.id]] += d * d * p * q

                for l in individual:
                    d2ll[ids[l.item.id]] += l.judge.precision

                # regularization terms
                for i,v in enumerate(d2ll):
                    d2ll[i] += len(ids) / (item_std * item_std) 
                    
                std = 1.0 / np.sqrt(d2ll)

                for item in items:
                    item.conf = 1.96 * std[ids[item.id]]

                actual = np.array([item.value for item in items])
                predicted = np.array([item.mean for item in items])

                r = spearmanr(actual,predicted)[0]
                if math.isnan(r):
                    run_accuracy.append(0.0)
                else:
                    run_accuracy.append(r)

            if i%2 is 0:
                r = FakeObject()
                r.left, r.right = pairwise_rate_fn(items)
                r.judge = random.choice(judges)
                r.value = r.judge.pairwise_decision_fn(r.judge, r.left, r.right)
                pairwise.append(r)
            else:
                r = FakeObject()
                r.item = individual_rate_fn(items)
                r.judge = random.choice(judges)
                r.value = r.judge.individual_decision_fn(r.judge, r.item)
                individual.append(r)

        for idx, v in enumerate(run_accuracy):
            accuracy[idx].append(v)

    return accuracy


if __name__ == "__main__":

    num_runs = 30 
    num_items = 100
    num_judges = 20 
    num_ratings = 401
    num_estimations = 10 

    estimation_mod = math.trunc(num_ratings / num_estimations)

    # individual
    #accuracy_ind = simulate_individual(num_runs, num_items, num_judges, num_ratings,
    #                                   estimation_mod, noisy_judgement, random_item)
    #acc_ind_mean = [np.mean(np.array(l)) for l in accuracy_ind]
    #acc_ind_lower = [acc_ind_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
    #                 enumerate(accuracy_ind)]
    #acc_ind_upper = [acc_ind_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
    #                 enumerate(accuracy_ind)]
    #plt.fill_between([i * estimation_mod for i in range(num_ratings)], acc_ind_lower,
    #                 acc_ind_upper, alpha=0.5, facecolor="blue")
    #plt.plot([i * estimation_mod for i in range(num_ratings)], acc_ind_mean,
    #         label="Individual", color="blue")

    # pairwise
    accuracy_pairwise = simulate_pairwise(num_runs, num_items, num_judges, num_ratings,
                                          estimation_mod, noisy_choice, random_pair)
    acc_pair_mean = [np.mean(np.array(l)) for l in accuracy_pairwise]
    acc_pair_lower = [acc_pair_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
                      enumerate(accuracy_pairwise)]
    acc_pair_upper = [acc_pair_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
                     enumerate(accuracy_pairwise)]
    plt.fill_between([i * estimation_mod for i in range(num_ratings)], acc_pair_lower,
                     acc_pair_upper, alpha=0.5, facecolor="green")
    plt.plot([i * estimation_mod for i in range(num_ratings)], acc_pair_mean,
             label="Pairwise", color="green")

    # combined 
    #accuracy_combined = simulate_combined(num_runs, num_items, num_judges,
    #                                      num_ratings, estimation_mod,
    #                       noisy_choice, random_pair, noisy_judgement, random_item)
    #acc_combined_mean = [np.mean(np.array(l)) for l in accuracy_combined]
    #acc_combined_lower = [acc_combined_mean[idx] - 1.96 * np.std(np.array(l)) for idx, l in
    #                  enumerate(accuracy_combined)]
    #acc_combined_upper = [acc_combined_mean[idx] + 1.96 * np.std(np.array(l)) for idx, l in
    #                 enumerate(accuracy_combined)]
    #plt.fill_between([i * estimation_mod for i in range(num_ratings)], acc_combined_lower,
    #                 acc_combined_upper, alpha=0.5, facecolor="red")
    #plt.plot([i * estimation_mod for i in range(num_ratings)], acc_combined_mean,
    #         label="Combined", color="red")

    plt.title("Simulated Accuracy for " + str(num_items) + " Items and " +
              str(num_judges) + " Judges (Avg of " + str(num_runs) + " Runs)")
    plt.xlabel("# of ratings")
    plt.ylabel("Spearman's Rank Correlation Coefficient")
    plt.legend(loc=4)
    plt.show()


