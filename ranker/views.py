from django.shortcuts import render
from ranker.models import Project, Item, Judge, Rating
from django.contrib.auth.models import User
#from sklearn import linear_model
# http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext, loader
from django.core.context_processors import csrf 
from django.core.urlresolvers import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib import messages
from ranker.forms import UserCreateForm, ProjectForm
from django.core.files.uploadedfile import SimpleUploadedFile
from scipy.optimize import fmin_bfgs, fmin_tnc, fmin_l_bfgs_b, check_grad
from django.db.models import Q
from math import exp, log
import numpy as np
import random

# regularization parameters
item_mean = 0.0
item_std = 10.0
judge_mean = 1.0
judge_std = 10.0

def register(request):
    if request.method == 'POST':
        form = UserCreateForm(request.POST)     # create form object
        if form.is_valid():
            form.save()
            messages.info(request, "Thank you for registering. You are now logged in.")
            new_user = authenticate(username=request.POST['username'],
                                    password=request.POST['password1'])
            login(request, new_user)
            return HttpResponseRedirect(reverse('dashboard'))
    else:
        form = UserCreateForm()

    args = {}
    args.update(csrf(request))
    args['form'] = form
    return render(request, 'ranker/register.html', args)


def index(request):
    template = loader.get_template('ranker/index.html')
    context = RequestContext(request, {})
    return HttpResponse(template.render(context))

@login_required
def dashboard(request):
    projects = Project.objects.filter(user=request.user)
    template = loader.get_template('ranker/dashboard.html')
    context = RequestContext(request, {'projects': projects})
    return HttpResponse(template.render(context))

@login_required
def create_project(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST, request.FILES)     # create form object

        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user
            instance.save()
            form.save_images()

            messages.success(request, "Project Created!")
            #return HttpResponseRedirect(reverse('dashboard'))
            return HttpResponseRedirect(reverse('view_project',
                                                kwargs={'project_id':
                                                        instance.id}))
    else:
        form = ProjectForm()

    args = {}
    args.update(csrf(request))
    args['form'] = form
    return render(request, 'ranker/create_project.html', args)

@login_required
def view_project(request, project_id):
    project = Project.objects.get(id=project_id)
    #update_model(project.id)
    #min_val = min([item.mean - item.conf for item in Item.objects.filter(project=project)])
    #max_val = max([item.mean + item.conf for item in Item.objects.filter(project=project)])
    template = loader.get_template('ranker/view_project.html')
    context = RequestContext(request, {'project': project})
                                      # 'min_val': min_val,
                                      # 'max_val': max_val})
    return HttpResponse(template.render(context))

def expz(val):
    if val > 12:
        val = 12
    elif val < -12:
        val = -12
    return exp(val)

def ll_2p(x, *args):
    ids = {i:idx for idx, i in enumerate(args[1])}
    jids = {i:idx + len(ids) for idx, i in enumerate(args[0])}
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    ll = 0.0
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        d = x[jids[r.judge.id]]

        y = r.value
        z = d * (left - right)
        ez = expz(z)
        ll += y * z - log(1 + ez)

    # Normal prior on means
    item_reg = 0.0
    for i in ids:
        diff = x[ids[i]] - item_mean
        item_reg += diff * diff
    item_reg = (-1.0 / (2 * item_std * item_std)) * item_reg

    # Normal prior on discriminations
    judge_reg = 0.0
    for i in jids:
        diff = x[jids[i]] - judge_mean
        judge_reg += diff * diff
    judge_reg = (-1.0 / (2 * judge_std * judge_std)) * judge_reg

    return -1.0 * (ll + item_reg + judge_reg)

def ll_2p_grad(x, *args):
    ids = {i:idx for idx, i in enumerate(args[1])}
    jids = {i:idx + len(ids) for idx, i in enumerate(args[0])}
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

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

    # Normal prior on means
    item_reg = np.array([0.0 for v in x])
    for i in ids:
        item_reg[ids[i]] += (x[ids[i]] - item_mean)
    item_reg = (-1.0 / (item_std * item_std)) * item_reg

    # Gamma prior on discriminations
    judge_reg = np.array([0.0 for v in x])
    for i in jids:
        judge_reg[jids[i]] += (x[jids[i]] - judge_mean)
    judge_reg = (-1.0 / (judge_std * judge_std)) * judge_reg

    return -1 * (grad + item_reg + judge_reg)
        
def ll_1p(x, *args):
    ids = {i:idx for idx, i in enumerate(args)}
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    ll = 0.0
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        y = r.value
        z = (left - right)
        ez = expz(z)
        ll += y * z - log(1 + ez)

    # Normal prior on means
    item_reg = 0.0
    for i in ids:
        diff = x[ids[i]] - item_mean
        item_reg += diff * diff
    item_reg = (-1.0 / (2 * item_std * item_std)) * item_reg

    return -1.0 * (ll + item_reg)

def ll_1p_grad(x, *args):
    ids = {i:idx for idx, i in enumerate(args)}
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

    grad = np.array([0.0 for v in x])
    for r in ratings:
        left = x[ids[r.left.id]]
        right = x[ids[r.right.id]]
        y = r.value
        p = 1.0 / (1.0 + expz(-1 * (left-right)))

        g = y - p 
        grad[ids[r.left.id]] += g
        grad[ids[r.right.id]] += -1 * g

    # Normal prior on means
    item_reg = np.array([0.0 for v in x])
    for i in ids:
        item_reg[ids[i]] += (x[ids[i]] - item_mean)
    item_reg = (-1.0 / (item_std * item_std)) * item_reg

    return -1 * (grad + item_reg)

#def invlogit(val):
#    return 1.0 / (1.0 + expz(val))

#def ll_plus_grad(x, *args):
#    ids = {i:idx for idx, i in enumerate(args)}
#    #print(ids)
#    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))
#
#    ll = 0.0
#    grad = [0.0 for v in x]
#    for r in ratings:
#        left = x[ids[r.left.id]]
#        right = x[ids[r.right.id]]
#        y = r.value
#        p = max(0.00001, min(0.99999, invlogit(left-right)))
#        ez = expz(left - right)
#
#        ll += (y * log(p) + (1 - y) * log(1-p))
#        g = ((y - p) * ez * p) / (1 - p)
#        grad[ids[r.left.id]] += g
#        grad[ids[r.right.id]] += -1 * g
#
#    return -1.0 * np.array(ll), -1.0 * np.array(grad)


# OLD, i think it returns the same parameters, but less precise. 
#def ll_1p(x, *args):
#    #print(args)
#    #args = args[0]
#    ids = {i:idx for idx, i in enumerate(args)}
#    #print(ids)
#    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))
#
#    ll = 0.0
#    for r in ratings:
#        left = x[ids[r.left.id]]
#        right = x[ids[r.right.id]]
#        y = r.value
#        p = max(0.00001, min(0.99999, invlogit(left-right)))
#        ll += (y * log(p) + (1 - y) * log(1-p))
#
#    return -1.0 * np.array(ll)
#
#def ll_1p_grad(x, *args):
#    #print(args)
#    #args = args[0]
#    ids = {i:idx for idx, i in enumerate(args)}
#    #print(ids)
#    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))
#
#    grad = [0.0 for v in x]
#    for r in ratings:
#        left = x[ids[r.left.id]]
#        right = x[ids[r.right.id]]
#        y = r.value
#        p = max(0.00001, min(0.99999, invlogit(left-right)))
#        ez = expz(-1(left - right))
#
#        g = ((y - p) * ez * p) / (1 - p)
#        grad[ids[r.left.id]] += g
#        grad[ids[r.right.id]] += -1 * g
#
#    return -1 * np.array(grad)

def update_model(project_id):
    project = Project.objects.get(id=project_id)
    items = [item for item in Item.objects.filter(project=project).order_by('id')]
    ids = [item.id for item in items]
    judges = [judge for judge in
              Judge.objects.filter(ratings__project=project).order_by('id')]
    jids = [judge.id for judge in judges]
    #x0 = [item.mean for item in items] + [float(judge.discrimination) for judge in judges]
    x0 = [0.0 for item in items] + [1.0 for judge in judges]

    ######## 1PL #############
    # BFGS
    #result = fmin_bfgs(ll_1p, x0, fprime=ll_1p_grad, args=tuple(ids),
    #                   disp=True)

    # Truncated Newton
    #bounds = [(-12,12) for v in x0]
    #result = fmin_tnc(ll_1p, x0, fprime=ll_1p_grad, args=tuple(ids), bounds=bounds,
    #                   disp=True)[0]
    ##########################

    ######## 2PL #############
    # BFGS
    result = fmin_bfgs(ll_2p, x0, fprime=ll_2p_grad, 
                       args=(tuple(jids), tuple(ids)), disp=False)

    # Truncated Newton
    #bounds = [(-12,12) for v in ids] + [(0,12) for v in jids]
    #result = fmin_tnc(ll_2p, x0, fprime=ll_2p_grad, 
    #                  args=(tuple(jids), tuple(ids)), bounds=bounds, disp=True)[0]
    ##########################

    #print(result)

    for idx, item in enumerate(items):
        item.mean = result[idx]
        item.save()

    if len(result) > len(ids):
        for idx, judge in enumerate(judges):
            judge.discrimination = result[idx + len(ids)]
            judge.save()

    # compute the stds
    d2ll = np.array([0.0 for item in items])

    ids = {i: idx for idx, i in enumerate(ids)}
    jids = {i: idx for idx, i in enumerate(jids)}

    for r in Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids)):
        left = float(r.left.mean)
        right = float(r.right.mean)
        d = float(r.judge.discrimination)
        p = 1.0 / (1.0 + expz(-1 * d * (left-right)))
        q = 1 - p
        d2ll[ids[r.left.id]] += d * d * p * q
        d2ll[ids[r.right.id]] += d * d * p * q

    # regularization terms
    for i,v in enumerate(d2ll):
        d2ll[i] += len(ids) / (item_std * item_std) + len(jids) / (judge_std * judge_std)

    std = 1.0 / np.sqrt(d2ll)

    for idx, item in enumerate(items):
        item.conf = 1.96 * std[idx]
        item.save()

def rate(request, project_id):
    project = Project.objects.get(id=project_id)

    item1 = None
    item2 = None
    diff = float('inf')
    i1 = None
    i2 = None
    for item in Item.objects.filter(project=project).order_by('mean'):
        i1 = i2
        i2 = item
        if i1 and i2 and abs(i1.mean - i2.mean) <= diff:
            item1 = i1
            item2 = i2
            diff = abs(i1.mean - i2.mean)

    if random.random() > 0.5:
        temp = item1
        item1 = item2
        item2 = temp
    
    # pick random then pick closest
    #item1 = Item.objects.filter(project=project).order_by('?')[0]
    #itemsG = Item.objects.filter(project=project,
    #                            mean__gte=item1.mean).exclude(id=item1.id).order_by('mean')
    #itemsL = Item.objects.filter(project=project,
    #                            mean__lte=item1.mean).exclude(id=item1.id).order_by('-mean')
    #if not itemsG:
    #    item2 = itemsL[0]
    #elif not itemsL:
    #    item2 = itemsG[0]
    #elif abs(item1.mean - itemsG[0].mean) < abs(item1.mean - itemsL[0].mean):
    #    item2 = itemsG[0]
    #else:
    #    item2 = itemsL[0]

    # pick 2 randomly
    #items = Item.objects.filter(project=project).order_by("?")[:2]
    template = loader.get_template('ranker/rate.html')
    context = RequestContext(request, {'project': project,
                                       'item1': item1,
                                       'item2': item2})
    return HttpResponse(template.render(context))

def vote(request, project_id, item1_id, item2_id, value):
    ip = request.META.get('REMOTE_ADDR')

    if not ip:
        ip = "127.0.0.1"

    judge, new = Judge.objects.get_or_create(ip_address=ip)

    item1 = Item.objects.get(id=item1_id)
    item2 = Item.objects.get(id=item2_id)

    rating = Rating(judge=judge, left=item1, right=item2, value=value,
                    project=item1.project)
    rating.save()

    update_model(project_id)

    return HttpResponseRedirect(reverse('rate', kwargs={'project_id':
                                                        project_id}))


@login_required
def update_project(request, project_id):
    project = Project.objects.get(id=project_id)
    if request.method == 'POST':
        form = ProjectForm(instance=project, data=request.POST, files=request.FILES)     # create form object

        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user
            instance.save()
            form.save_images()

            messages.success(request, "Project Updated!")
            return HttpResponseRedirect(reverse('view_project',
                                                kwargs={'project_id':
                                                        instance.id}))
    else:
        form = ProjectForm(instance=project)

    args = {}
    args.update(csrf(request))
    args['form'] = form
    args['project_id'] = project_id
    return render(request, 'ranker/update_project.html', args)

@login_required
def delete_project(request, project_id):
    project = Project.objects.get(pk=project_id)
    project.delete()
    messages.success(request, "Project Deleted!")
    return HttpResponseRedirect(reverse('dashboard'))
    
@login_required
def export_rankings(request):
    pass

@login_required
def export_ratings(request):
    pass

