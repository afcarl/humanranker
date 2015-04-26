from math import exp, log, sqrt
import numpy as np
import random
import csv
from hashlib import sha1
from scipy.optimize import fmin_tnc
from django.db.models import Q
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext, loader
from django.core.context_processors import csrf 
from django.core.urlresolvers import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.shortcuts import render
#from django.core.files.uploadedfile import SimpleUploadedFile
#from django.contrib.auth.models import User
#from sklearn import linear_model
# http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
from ranker.models import Project
from ranker.models import Item
from ranker.models import Judge
from ranker.models import Rating
from ranker.models import Likert
from ranker.forms import UserCreateForm
from ranker.forms import ProjectForm
from ranker.forms import ProjectUpdateForm

# regularization parameters
item_mean = 0.0
item_std = 1.0

discrim_mean = 1.0
discrim_std = 1.0

bias_mean = 4.0
bias_std = 1.5

#noise_mean = 1.0
#noise_std = 1 - sqrt(0.5) # produces a discrimination of mean 1 std 1

prec_mean = 1.0
prec_std = 1.0

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

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    update_model(project_id)
    judges = Judge.objects.filter(project=project).distinct()
    template = loader.get_template('ranker/view_project.html')
    context = RequestContext(request, {'project': project,
                                       'judges': judges})

    return HttpResponse(template.render(context))

def expz(val):
    if val > 12:
        val = 12
    elif val < -12:
        val = -12
    return exp(val)

def ll_combined(x, *args):
    ids = {i:idx for idx, i in enumerate(args[0])}
    discids = {i:idx + len(ids) for idx, i in enumerate(args[1])}
    biasids = {i:idx + len(ids) + len(discids) for idx, i in enumerate(args[1])}
    precids = {i:idx + len(ids) + 2*len(discids) for idx, i in enumerate(args[1])}
    ratings = args[2]
    likerts = args[3]
    #ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids))

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
        n = sqrt(1/p)

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

def update_model(project_id):
    project = Project.objects.get(id=project_id)
    items = Item.objects.filter(project=project).order_by('id').distinct()
    ids = [item.id for item in items]
    judges = Judge.objects.filter(project=project).order_by('id').distinct()
    jids = [judge.id for judge in judges]
    ratings = Rating.objects.filter(Q(left__in=ids)|Q(right__in=ids)).distinct()
    likerts = Likert.objects.filter(item__in=ids).distinct()

    x0 = [item.mean for item in items] 
    x0 += [judge.discrimination for judge in judges]
    x0 += [judge.bias for judge in judges]
    x0 += [judge.precision for judge in judges]

    #print(x0)
    #from scipy.optimize import check_grad, approx_fprime
    #print(check_grad(ll_combined, ll_combined_grad, x0, tuple(ids), tuple(jids),
    #                                         ratings, likerts))
    #print(approx_fprime(x0, ll_combined, sqrt(np.finfo(float).eps), tuple(ids), tuple(jids), ratings,
    #              likerts))
    #print(ll_combined_grad(x0,  tuple(ids), tuple(jids), ratings, likerts)) 

    bounds = [('-inf','inf') for v in ids]
    bounds += [(0.001,'inf') for v in jids]
    bounds += [('-inf','inf') for v in jids]
    bounds += [(0.001,'inf') for v in jids]

    result = fmin_tnc(ll_combined, x0, 
                      #approx_grad=True,
                      fprime=ll_combined_grad, 
                      args=(tuple(ids), tuple(jids), ratings, likerts),
                      bounds=bounds,
                      disp=False)[0]

    ######## 2PL #############
    #x0 = [item.mean for item in items] + [judge.discrimination for judge in judges]
    ##x0 = [2 * (random.random() - 0.5) for item in items] + [3 * random.random() for judge in judges]
    ##x0 = [0.0 for item in items] + [1.0 for judge in judges]

    ## Check the gradient calculation
    #print(check_grad(ll_2p, ll_2p_grad, x0, tuple(ids), tuple(jids),
    #                                         ratings))

    ## Truncated Newton
    #bounds = [('-inf','inf') for v in ids] + [(0.001,'inf') for v in jids]
    #result = fmin_tnc(ll_2p, x0, 
    #                  #approx_grad=True,
    #                  fprime=ll_2p_grad, 
    #                  args=(tuple(jids), tuple(ids), ratings), bounds=bounds,
    #                  disp=False)[0]
    ##########################
    #print(result)

    ids = {i: idx for idx, i in enumerate(ids)}
    discids = {i: idx + len(ids) for idx, i in enumerate(jids)}
    biasids = {i: idx + len(ids) + len(jids) for idx, i in enumerate(jids)}
    precids = {i: idx + len(ids) + 2 * len(jids) for idx, i in enumerate(jids)}

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
        
        d2ll[i] += (len(discids) / (discrim_std * discrim_std))

        d2ll[i] += (len(biasids) / (bias_std * bias_std))

        d2ll[i] += (len(precids) / (prec_std * prec_std))
    #print(d2ll)

    std = 1.0 / np.sqrt(d2ll)
    #print(std)

    for item in items:
        item.conf = 1.96 * std[ids[item.id]]
        item.save()

def random_pair(project):
    """
    Returns a random pair of items from the given project.
    """
    items = list(Item.objects.filter(project=project).distinct().order_by("?")[:2])
    return items[0], items[1]

def conf_adjacent_pair(project):
    """
    Returns an adjacent pair of items that have the most confidence interval
    overlap.
    """
    item1 = None
    item2 = None
    diff = float('-inf')
    i1 = None
    i2 = None
    for item in Item.objects.filter(project=project).order_by('mean').distinct():
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

def likert(request, project_id):
    project = Project.objects.get(id=project_id)

    ip = request.META.get('REMOTE_ADDR')
    if not ip:
        ip = "127.0.0.1"

    judge = Judge.objects.filter(ip_address=ip, project=project).first()

    items = Item.objects.filter(project=project).distinct()
    if judge:
        h = judge.get_hashkey()
        count = len(judge.likerts.all())
        last = Likert.objects.filter(project=project,judge=judge).order_by('-added').first()
        items = items.exclude(id=last.item.id)
    else:
        ip_pid = str(ip) + "-" + str(project.id)
        h = sha1(ip_pid.encode('utf-8')).hexdigest()[0:10]
        count = 0

    item = items.order_by("?")[0]

    template = loader.get_template('ranker/likert.html')
    context = RequestContext(request, {'project': project,
                                       'item': item,
                                       'key': h,
                                       'count': count})
    return HttpResponse(template.render(context))


def rate(request, project_id):
    project = Project.objects.get(id=project_id)
    item1, item2 = random_pair(project)

    ip = request.META.get('REMOTE_ADDR')
    if not ip:
        ip = "127.0.0.1"

    judge = Judge.objects.filter(ip_address=ip, project=project).first()

    if judge:
        h = judge.get_hashkey()
        count = len(judge.ratings.all())
    else:
        ip_pid = str(ip) + "-" + str(project.id)
        h = sha1(ip_pid.encode('utf-8')).hexdigest()[0:10]
        count = 0

    template = loader.get_template('ranker/rate.html')
    context = RequestContext(request, {'project': project,
                                       'item1': item1,
                                       'item2': item2,
                                       'key': h,
                                       'count': count})
    return HttpResponse(template.render(context))

def vote_likert(request, project_id, item_id, value):
    ip = request.META.get('REMOTE_ADDR')

    if not ip:
        ip = "127.0.0.1"

    project = Project.objects.get(id=project_id)
    judge, new = Judge.objects.get_or_create(ip_address=ip,project=project)
    item = Item.objects.get(id=item_id)

    likert = Likert(judge=judge, item=item, value=value,
                    project=item.project)
    likert.save()

    #update_model(project_id)

    return HttpResponseRedirect(reverse('likert', kwargs={'project_id':
                                                        project_id}))

def vote(request, project_id, item1_id, item2_id, value):
    ip = request.META.get('REMOTE_ADDR')

    if not ip:
        ip = "127.0.0.1"

    project = Project.objects.get(id=project_id)
    judge, new = Judge.objects.get_or_create(ip_address=ip,project=project)

    item1 = Item.objects.get(id=item1_id)
    item2 = Item.objects.get(id=item2_id)

    rating = Rating(judge=judge, left=item1, right=item2, value=value,
                    project=item1.project)
    rating.save()

    #update_model(project_id)

    return HttpResponseRedirect(reverse('rate', kwargs={'project_id':
                                                        project_id}))


@login_required
def update_project(request, project_id):
    project = Project.objects.get(id=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    if request.method == 'POST':
        form = ProjectUpdateForm(instance=project, data=request.POST, files=request.FILES)     # create form object

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
        form = ProjectUpdateForm(instance=project)

    args = {}
    args.update(csrf(request))
    args['form'] = form
    args['project_id'] = project_id
    return render(request, 'ranker/update_project.html', args)

@login_required
def delete_project(request, project_id):
    project = Project.objects.get(pk=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    project.delete()
    messages.success(request, "Project Deleted!")
    return HttpResponseRedirect(reverse('dashboard'))

@login_required
def view_item(request, item_id):
    item = Item.objects.get(pk=item_id)

    if not request.user == item.project.user:
        messages.error(request, "Sorry! You do not have permission to view this item.")
        return HttpResponseRedirect(reverse('dashboard'))

    args = {}
    args.update(csrf(request))
    args['item'] = item
    return render(request, 'ranker/view_item.html', args)

@login_required
def delete_item(request, item_id):
    item = Item.objects.get(pk=item_id)

    if not request.user == item.project.user:
        messages.error(request, "Sorry! You do not have permission to view this item.")
        return HttpResponseRedirect(reverse('dashboard'))

    project_id = item.project.id
    item.delete()
    messages.success(request, "Item Deleted!")
    return HttpResponseRedirect(reverse('view_project',
                                        kwargs={'project_id':
                                               project_id}))

@login_required
def export_judge_estimates(request, project_id):
    project = Project.objects.get(id=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    judges = Judge.objects.filter(project=project).distinct()

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="judge-estimate-export-' + str(project.id) + '.csv"'

    writer = csv.writer(response)
    writer.writerow(['id', 'hash key', '# pairwise ratings', 
                     'pairwise discrimination', '# individual ratings',
                     'mean individual rating', 'individual discrimination'])

    for judge in judges:
        writer.writerow([judge.id, judge.get_hashkey(),
                         len(judge.ratings.all()), judge.discrimination,
                         len(judge.likerts.all()), judge.bias,
                         judge.precision])

    return response
    
@login_required
def export_rankings(request, project_id):
    project = Project.objects.get(id=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    items = Item.objects.filter(project=project).distinct()

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="item-estimate-export-' + str(project.id) + '.csv"'

    writer = csv.writer(response)
    writer.writerow(['id', 'name', 'parameter estimate', 
                     '+/- 95% confidence interval'])
    for item in items:
        writer.writerow([item.id, item.name, item.mean, item.conf])

    return response

@login_required
def export_ratings(request, project_id):
    project = Project.objects.get(id=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    ratings = Rating.objects.filter(project=project).distinct()

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="pairwise-export-' + str(project.id) + '.csv"'

    writer = csv.writer(response)
    writer.writerow(['rating_id', 'left_item_id', 'right_item_id',
                     'left_item_name', 'right_item_name',
                     'rating_value (left=1)', 'judge_id', 'judge_hash'])
    for r in ratings:
        writer.writerow([r.id, r.left.id, r.right.id, r.left.name,
                         r.right.name, r.value, r.judge.id,
                         r.judge.get_hashkey()])

    return response

@login_required
def export_likerts(request, project_id):
    project = Project.objects.get(id=project_id)

    if not request.user == project.user:
        messages.error(request, "Sorry! You do not have permission to view this project.")
        return HttpResponseRedirect(reverse('dashboard'))

    likerts = Likert.objects.filter(project=project).distinct()

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="individual-export-' + str(project.id) + '.csv"'

    writer = csv.writer(response)
    writer.writerow(['likert_id', 'item_id', 'item_name', 
                     'rating_value (5=strongly agree, 1=strongly disagree)', 
                     'judge_id', 'judge_hash'])
    for l in likerts:
        writer.writerow([l.id, l.item.id, l.item.name, 
                         l.value, 
                         l.judge.id,
                         l.judge.get_hashkey()])

    return response
