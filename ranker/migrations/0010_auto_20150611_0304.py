# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0009_auto_20150426_1536'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='judge',
            options={'ordering': ['-discrimination', '-precision']},
        ),
        migrations.AddField(
            model_name='project',
            name='likert_mean',
            field=models.FloatField(default=4.0),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='project',
            name='likert_scale',
            field=models.FloatField(default=1.0),
            preserve_default=True,
        ),
    ]
