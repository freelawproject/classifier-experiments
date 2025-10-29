from django.db import models


class DocketEntry(models.Model):
    recap_id = models.IntegerField()
