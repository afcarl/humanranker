from math import sqrt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from django.db.models import Q

from pairwise.celery import app
from ranker.models import Project
from ranker.models import Item
from ranker.models import Judge
from ranker.models import Rating
from ranker.models import Likert
from ranker.estimator import ll_combined
from ranker.estimator import ll_combined_grad
from ranker.estimator import expz
from ranker.estimator import item_mean
from ranker.estimator import item_std
from ranker.estimator import discrim_mean
from ranker.estimator import bias_mean
from ranker.estimator import prec_mean

@app.task
def update_model(project_id):
    project = Project.objects.get(id=project_id)
    items = Item.objects.filter(project=project).order_by('id').distinct()
    ids = [item.id for item in items]
    judges = Judge.objects.filter(project=project).order_by('id').distinct()
    jids = [judge.id for judge in judges]
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids)).distinct()
    likerts = Likert.objects.filter(item__in=ids).distinct()

    #x0 = [item.mean for item in items] 
    #x0 += [judge.discrimination for judge in judges]
    #x0 += [judge.bias for judge in judges]
    #x0 += [judge.precision for judge in judges]
    x0 = [item_mean for item in items] 
    x0 += [discrim_mean for judge in judges]
    x0 += [bias_mean for judge in judges]
    x0 += [prec_mean for judge in judges]

    # The parameters linking likert to pairwise (i.e., overall likert mean)
    x0 += [1.0, 4.0]

    #print(x0)
    #from scipy.optimize import check_grad, approx_fprime
    #from math import sqrt
    #print(check_grad(ll_combined, ll_combined_grad, x0, tuple(ids), tuple(jids),
    #                                         ratings, likerts))
    #print(approx_fprime(x0, ll_combined, sqrt(np.finfo(float).eps), tuple(ids), tuple(jids), ratings,
    #              likerts))
    #print(ll_combined_grad(x0,  tuple(ids), tuple(jids), ratings, likerts)) 

    bounds = [('-inf','inf') for v in ids]
    bounds += [(0.001,'inf') for v in jids]
    bounds += [('-inf','inf') for v in jids]
    bounds += [(0.001,'inf') for v in jids]

    # bounds on likert-pairwise link parameters (i.e, likert mean)
    bounds += [(0.001, 'inf'), (1.0, 7.0)]

    result = fmin_l_bfgs_b(ll_combined, x0, 
                      #approx_grad=True,
                      fprime=ll_combined_grad, 
                      args=(tuple(ids), tuple(jids), ratings, likerts),
                      bounds=bounds,
                      disp=False)[0]

    ids = {i: idx for idx, i in enumerate(ids)}
    discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
    biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
    precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}

    project.likert_mean = result[-1]
    project.likert_scale = 1 / sqrt(result[-2])
    project.save()

    for item in items:
        item.mean = result[ids[item.id]]
        item.conf = 10000.0
        item.save()

    for judge in judges:
        judge.discrimination = result[discids[judge.id]]
        judge.bias = result[biasids[judge.id]]
        judge.precision = result[precids[judge.id]]
        judge.save()

    # compute the stds
    d2ll = np.array([0.0 for item in items])

    for r in ratings:
        d = r.judge.discrimination
        left = r.left.mean
        right = r.right.mean
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))
        q = 1 - p
        d2ll[ids[r.left.id]] += d * d * p * q
        d2ll[ids[r.right.id]] += d * d * p * q
    
    for l in likerts:
        d2ll[ids[l.item.id]] += l.judge.precision

    # regularization terms
    for i,v in enumerate(d2ll):
        d2ll[i] += len(ids) / (item_std * item_std) 

    std = 1.0 / np.sqrt(d2ll)
    #print(std)

    for item in items:
        item.conf = 1.96 * std[ids[item.id]]
        item.save()

@app.task
def async_test(x):
    print(x)
