from django.conf import settings
from django.contrib.postgres.operations import TrigramExtension
from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):

    dependencies = []

    operations = [
        TrigramExtension(),
        VectorExtension(),
        migrations.RunSQL(f"ALTER DATABASE {settings.DATABASES['default']['NAME']} SET hnsw.ef_search = 200;"),
    ]
