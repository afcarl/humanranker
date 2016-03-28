# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0002_auto_20150401_2236'),
    ]

    operations = [
        migrations.AlterField(
            model_name='item',
            name='conf',
            field=models.FloatField(default=10000),
            preserve_default=True,
        ),
    ]
