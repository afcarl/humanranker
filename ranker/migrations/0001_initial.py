# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import ranker.models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
                ('image', models.ImageField(upload_to=ranker.models.Item.fancy_path)),
                ('mean', models.FloatField(default=0.0)),
                ('conf', models.FloatField(default=0.0)),
                ('added', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-mean'],
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Judge',
            fields=[
                ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True, serialize=False)),
                ('ip_address', models.GenericIPAddressField(unique=True)),
                ('discrimination', models.FloatField(default=1.0)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
                ('prompt', models.TextField()),
                ('added', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Rating',
            fields=[
                ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True, serialize=False)),
                ('value', models.FloatField()),
                ('added', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('judge', models.ForeignKey(to='ranker.Judge', related_name='ratings')),
                ('left', models.ForeignKey(to='ranker.Item', related_name='left_ratings')),
                ('project', models.ForeignKey(to='ranker.Project', related_name='ratings')),
                ('right', models.ForeignKey(to='ranker.Item', related_name='right_ratings')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='item',
            name='project',
            field=models.ForeignKey(to='ranker.Project', related_name='items'),
            preserve_default=True,
        ),
    ]
