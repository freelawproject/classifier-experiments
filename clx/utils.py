import os

import django


def init_django():
    os.environ["DJANGO_SETTINGS_MODULE"] = "clx.settings"
    django.setup()
