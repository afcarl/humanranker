import random
import csv
import numpy as np

from hashlib import sha1
from itertools import combinations
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
from ranker.estimator import ll_combined
from ranker.estimator import ll_combined_grad
from ranker.estimator import expz
from ranker.estimator import item_mean
from ranker.estimator import item_std
from ranker.estimator import discrim_mean
from ranker.estimator import discrim_std
from ranker.estimator import bias_mean
from ranker.estimator import bias_std
from ranker.estimator import prec_mean
from ranker.estimator import prec_std

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

        #these shouldn't get included in item parameter estimates.
        #d2ll[i] += (len(discids) / (discrim_std * discrim_std))
        #d2ll[i] += (len(biasids) / (bias_std * bias_std))
        #d2ll[i] += (len(precids) / (prec_std * prec_std))
    #print(d2ll)

    std = 1.0 / np.sqrt(d2ll)
    #print(std)

    for item in items:
        item.conf = 1.96 * std[ids[item.id]]
        item.save()

def random_pair(project, judge=None):
    """
    Returns a random pair of items from the given project.
    """
    ids = [i.id for i in Item.objects.filter(project=project)]
    pairs = {p for p in combinations(ids,2)}

    if judge:
        for r in Rating.objects.filter(judge=judge, project=project):
            if (r.left.id, r.right.id) in pairs:
                pairs.remove((r.left.id, r.right.id))
            elif (r.right.id, r.left.id) in pairs:
                pairs.remove((r.right.id, r.left.id))

    pairs = list(pairs)

    if len(pairs) > 0:
        random.shuffle(pairs)

        pair = pairs[0]
        item1 = Item.objects.get(id=pair[0])
        item2 = Item.objects.get(id=pair[1])

        if random.random() > 0.5:
            temp = item1
            item1 = item2
            item2 = temp

        return item1, item2
    else:
        return None, None

    #items = list(Item.objects.filter(project=project).distinct().order_by("?")[:2])
    #return items[0], items[1]

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
        done = [l.item.id for l in Likert.objects.filter(project=project,judge=judge)]
        items = items.exclude(id__in=done).distinct()
        #last = Likert.objects.filter(project=project,judge=judge).order_by('-added').first()
        #items = items.exclude(id=last.item.id)
    else:
        ip_pid = str(ip) + "-" + str(project.id)
        h = sha1(ip_pid.encode('utf-8')).hexdigest()[0:10]
        count = 0

    item = items.order_by("?").first()

    if item:
        template = loader.get_template('ranker/likert.html')
    else:
        template = loader.get_template('ranker/done.html')

    context = RequestContext(request, {'project': project,
                                       'item': item,
                                       'key': h,
                                       'count': count})
    return HttpResponse(template.render(context))

def rate(request, project_id):
    project = Project.objects.get(id=project_id)

    ip = request.META.get('REMOTE_ADDR')
    if not ip:
        ip = "127.0.0.1"

    judge = Judge.objects.filter(ip_address=ip, project=project).first()

    if judge:
        h = judge.get_hashkey()
        count = len(judge.ratings.all())
        item1, item2 = random_pair(project, judge)
    else:
        ip_pid = str(ip) + "-" + str(project.id)
        h = sha1(ip_pid.encode('utf-8')).hexdigest()[0:10]
        count = 0
        item1, item2 = random_pair(project)

    if item1 is not None:
        template = loader.get_template('ranker/rate.html')
    else:
        template = loader.get_template('ranker/done.html')

    context = RequestContext(request, {'project': project,
                                       'item1': item1,
                                       'item2': item2,
                                       'key': h,
                                       'count': count})
    return HttpResponse(template.render(context))

def vote_likert(request, project_id):

    if (request.method == 'POST' and 'item_id' in request.POST and 'rating' in
        request.POST):

        item_id = request.POST['item_id']
        value = request.POST['rating']

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
    else:
        messages.error(request, "Error processing your rating. Try again.")

    return HttpResponseRedirect(reverse('likert', kwargs={'project_id':
                                                        project_id}))

def vote(request, project_id):
    
    if (request.method == 'POST' and 'item1_id' in request.POST and 'item2_id'
        in request.POST and 'rating' in request.POST):

        item1_id = request.POST['item1_id']
        item2_id = request.POST['item2_id']
        value = request.POST['rating']

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
    else:
        messages.error(request, "Error processing your rating. Try again.")

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
                     'rating_value', 'judge_id', 'judge_hash'])
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
                     'rating_value', 
                     'judge_id', 'judge_hash'])
    for l in likerts:
        writer.writerow([l.id, l.item.id, l.item.name, 
                         l.value, 
                         l.judge.id,
                         l.judge.get_hashkey()])

    return response
