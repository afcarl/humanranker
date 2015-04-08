import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

class Foo:
    pass

def noisy_evaluate(judge, item, error):
    rating = judge.bias + random.normalvariate(0, judge.discrimination)
    return min(10, max(1,random.normalvariate(rating, error)))
 
def simulate(num_runs, num_items, num_judges, num_ratings, decision_fn,
             error):
    reliability = np.array([0.0 for i in range(num_ratings)])

    for run in range(num_runs):
        items = []
        ids = []
        for i in range(num_items):
            item = Foo()
            item.id = i
            #item.value = random.normalvariate(0,1)
            item.value = random.uniform(-10,10)
            items.append(item)
            ids.append(i)

        judges = []
        jids = []
        for i in range(num_judges):
            judge = Foo()
            judge.id = i
            #judge.value = max(0.001, random.normalvariate(0,1))
            judge.bias = random.uniform(0,10)
            judge.discrimination = max(0, random.uniform(1,1))
            judge.decision_fn = decision_fn
            judges.append(judge)
            jids.append(i)

        # generate ratings
        ratings = []
        run_reliability = []
        for i in range(num_ratings):
            if len(ratings) > 0:
                v = DictVectorizer()
                X = [x for x,y in ratings]
                y = [y for x,y in ratings]
                X = v.fit_transform(X).toarray()
                y = np.array(y)
                print(X, y)

                clf = Ridge()
                clf.fit(X,y)

                print(clf.coef_)
                print(v.inverse_transform(clf.coef_))
            else:
                run_reliability.append(0)

            #r = spearmanr(actual,predicted)[0]

            r = {}
            item = random.choice(items)
            judge = random.choice(judges)
            for i in items:
                r['item' + str(i.id)] = 0
            for j in judges:
                r['judge' + str(j.id)] = 0
            r['item' + str(item.id)] = 1
            r['judge' + str(judge.id)] = 1
            r = (r, judge.decision_fn(judge, item, error))
            ratings.append(r)

    return 0.0

if __name__ == "__main__":

    num_runs = 1
    num_items = 40
    num_judges = 5
    num_ratings = 2
    error = 1

    reliability_random = simulate(num_runs, num_items, num_judges, num_ratings,
                           noisy_evaluate, error)


    plt.plot([i for i in range(num_ratings)], reliability_random, label="Random Pairs")
#    plt.plot([i for i in range(num_ratings)], reliability_conf,
#             label="Overlapping Confidence Pairs")
    plt.title("Simulated Reliability for " + str(num_items) + " Items and " +
              str(num_judges) + " Judges (Avg of " + str(num_runs) + " Runs)")
    plt.xlabel("# of pairwise comparisons")
    plt.ylabel("Spearman's Rank Correlation Coefficient")
    plt.legend(loc=4)
    plt.show()
