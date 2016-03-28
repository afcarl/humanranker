# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ranker', '0005_auto_20150425_1539'),
    ]

    operations = [
        migrations.CreateModel(
            name='Likert',
            fields=[
                ('id', models.AutoField(serialize=False, verbose_name='ID', primary_key=True, auto_created=True)),
                ('value', models.FloatField()),
                ('added', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('item', models.ForeignKey(to='ranker.Item', related_name='likerts')),
                ('judge', models.ForeignKey(to='ranker.Judge', related_name='likerts')),
                ('project', models.ForeignKey(to='ranker.Project', related_name='likerts')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AlterModelOptions(
            name='judge',
            options={'ordering': ['-pairwise_discrimination']},
        ),
        migrations.RenameField(
            model_name='judge',
            old_name='discrimination',
            new_name='pairwise_discrimination',
        ),
        migrations.AddField(
            model_name='project',
            name='individual_prompt',
            field=models.TextField(default='A default likert statement.'),
            preserve_default=False,
        ),
    ]
