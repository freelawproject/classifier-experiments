from django.contrib.postgres.operations import TrigramExtension
from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):

    dependencies = []

    operations = [
        TrigramExtension(),
        VectorExtension(),
    ]
