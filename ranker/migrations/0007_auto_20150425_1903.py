# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0006_auto_20150425_1601'),
    ]

    operations = [
        migrations.AddField(
            model_name='judge',
            name='individual_bias',
            field=models.FloatField(default=3.0),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='judge',
            name='individual_noise',
            field=models.FloatField(default=1.0),
            preserve_default=True,
        ),
    ]
