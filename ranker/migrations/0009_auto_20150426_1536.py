# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0008_auto_20150425_2101'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='likert_label_1',
            field=models.CharField(max_length=200, default='Strongly disagree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_2',
            field=models.CharField(max_length=200, default='Disagree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_3',
            field=models.CharField(max_length=200, default='Somewhat disagree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_4',
            field=models.CharField(max_length=200, default='Neither agree nor disagree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_5',
            field=models.CharField(max_length=200, default='Somewhat agree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_6',
            field=models.CharField(max_length=200, default='Agree'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_label_7',
            field=models.CharField(max_length=200, default='Strongly agree'),
            preserve_default=False,
        ),
    ]
