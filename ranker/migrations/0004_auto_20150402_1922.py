# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0003_auto_20150402_0525'),
    ]

    operations = [
        migrations.AddField(
            model_name='judge',
            name='project',
            field=models.ForeignKey(to='ranker.Project', default=1),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='judge',
            name='ip_address',
            field=models.GenericIPAddressField(),
            preserve_default=True,
        ),
        migrations.AlterUniqueTogether(
            name='judge',
            unique_together=set([('ip_address', 'project')]),
        ),
    ]
