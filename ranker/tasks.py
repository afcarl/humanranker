from datetime import datetime

import numpy as np
from scipy.optimize import basinhopping

from django.db.models import Q

from pairwise.celery import app
from ranker.models import Project
from ranker.models import Item
from ranker.models import Judge
from ranker.models import Rating
from ranker.models import Likert
from ranker.estimator import ll_combined
from ranker.estimator import ll_combined_grad
from ranker.estimator import item_prec
from ranker.estimator import BoundedStepper
from ranker.estimator import invlogit

@app.task
def update_model(project_id):
    project = Project.objects.get(id=project_id)
    last_likert = Likert.objects.filter(project=project).order_by('-added').first()
    last_rating = Rating.objects.filter(project=project).order_by('-added').first()
   
    if ((not last_likert or project.last_model_estimation > last_likert.added) and
        (not last_rating or project.last_model_estimation > last_rating.added)):
        return

    project.last_mode_estimation = datetime.now()
    project.save()

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
    x0 = [item.mean for item in items] 
    x0 += [judge.discrimination for judge in judges]
    x0 += [judge.bias for judge in judges]
    x0 += [judge.precision for judge in judges]

    # The parameters linking likert to pairwise (i.e., overall likert mean)
    x0 += [project.likert_mean, project.likert_scale]

    #print(x0)
    #from scipy.optimize import check_grad, approx_fprime
    #from math import sqrt
    #print(check_grad(ll_combined, ll_combined_grad, x0, tuple(ids), tuple(jids),
    #                                         ratings, likerts))
    #print(approx_fprime(x0, ll_combined, sqrt(np.finfo(float).eps), tuple(ids), tuple(jids), ratings,
    #              likerts))
    #print(ll_combined_grad(x0,  tuple(ids), tuple(jids), ratings, likerts)) 

    bounds = [('-inf','inf') for v in ids]
    bounds += [('-inf','inf') for v in jids]
    bounds += [('-inf','inf') for v in jids]
    bounds += [(0.001,'inf') for v in jids]
    bounds += [(0.001, 'inf'), ('-inf', 'inf')]

    stepper = BoundedStepper(bounds, 20)
    result = basinhopping(ll_combined, x0, disp=False, T=15, 
                          niter=10000, niter_success=3, take_step=stepper,
                          minimizer_kwargs={'method': 'TNC', 'args':
                                            (tuple(ids), tuple(jids), ratings,
                                             likerts),
                                            'jac': ll_combined_grad, 'bounds':
                                            bounds, 'options': {
                                                              'maxiter': 10000},
                                           })['x']
    #result = fmin_l_bfgs_b(ll_combined, x0, 
    #                  #approx_grad=True,
    #                  fprime=ll_combined_grad, 
    #                  args=(tuple(ids), tuple(jids), ratings, likerts),
    #                  bounds=bounds,
    #                  disp=False)[0]

    ids = {i: idx for idx, i in enumerate(ids)}
    discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
    biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
    precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}

    project.likert_mean = result[-1]
    project.likert_scale = result[-2]
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
        p = invlogit(d * (left - right))
        q = 1 - p
        d2ll[ids[r.left.id]] += d * d * p * q
        d2ll[ids[r.right.id]] += d * d * p * q
    
    for l in likerts:
        d2ll[ids[l.item.id]] += l.judge.precision

    # regularization terms
    for i,v in enumerate(d2ll):
        d2ll[i] += len(ids) * item_prec

    std = 1.0 / np.sqrt(d2ll)
    #print(std)

    for item in items:
        item.conf = 1.96 * std[ids[item.id]]
        item.save()

@app.task
def async_test(x):
    print(x)
