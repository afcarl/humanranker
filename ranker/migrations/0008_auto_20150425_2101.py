# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0007_auto_20150425_1903'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='judge',
            options={'ordering': ['-discrimination']},
        ),
        migrations.RenameField(
            model_name='judge',
            old_name='individual_bias',
            new_name='bias',
        ),
        migrations.RenameField(
            model_name='judge',
            old_name='pairwise_discrimination',
            new_name='discrimination',
        ),
        migrations.RenameField(
            model_name='judge',
            old_name='individual_noise',
            new_name='precision',
        ),
        migrations.RenameField(
            model_name='project',
            old_name='individual_prompt',
            new_name='individual_likert_prompt',
        ),
    ]
