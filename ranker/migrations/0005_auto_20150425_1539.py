# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0004_auto_20150402_1922'),
    ]

    operations = [
        migrations.RenameField(
            model_name='project',
            old_name='prompt',
            new_name='pairwise_prompt',
        ),
    ]
