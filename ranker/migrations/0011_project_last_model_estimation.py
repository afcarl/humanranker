# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import datetime


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0010_auto_20150611_0304'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='last_model_estimation',
            field=models.DateTimeField(default=datetime.datetime.now),
            preserve_default=True,
        ),
    ]
