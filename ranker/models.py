from django.db import models
from django.contrib.auth.models import User
#from django.db.models.signals import post_save
#from django.dispatch import receiver
from hashlib import sha1

#class UserProfile(models.Model):
#    discrimination = models.DecimalField(decimal_places=10, max_digits=20, default=1.0)
#    user = models.ForeignKey(User, unique=True, related_name="profile")
#
#@receiver(post_save, sender=User)
#def create_profile(sender, instance, **kwargs):
#    profile, new = UserProfile.objects.get_or_create(user=instance)

class Project(models.Model):
    name = models.CharField(max_length=200)
    pairwise_prompt = models.TextField()
    user = models.ForeignKey(User)
    added = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.name)

class Judge(models.Model):
    ip_address = models.GenericIPAddressField()
    project = models.ForeignKey(Project)
    discrimination = models.FloatField(default=1.0)

    class Meta:
        ordering = ['-discrimination']
        unique_together = ('ip_address', 'project',)

    def get_hashkey(self):
        ip_pid = str(self.ip_address) + "-" + str(self.project.id)
        return sha1(ip_pid.encode('utf-8')).hexdigest()[0:10]

class Item(models.Model):
    def fancy_path(self, filename):
            return 'images/%s/%s' % (self.project.id, filename)

    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to=fancy_path)
    mean = models.FloatField(default=0.0)
    #std = models.FloatField(default=0.0)
    conf = models.FloatField(default=10000)
    project = models.ForeignKey(Project, related_name="items")
    added = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str((self.id, self.name, self.mean))

    class Meta:
        ordering = ['-mean']

    def __add__(self, x):
        return self.mean + x

class Rating(models.Model):
    judge = models.ForeignKey(Judge, related_name='ratings')
    left = models.ForeignKey(Item, related_name='left_ratings')
    right = models.ForeignKey(Item, related_name='right_ratings')
    value = models.FloatField() # 1 = left, 0 = right
    added = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    project = models.ForeignKey(Project, related_name="ratings")

    def __str__(self):
        return str((self.left, self.right, self.value))



