from django.contrib import admin
from home.models import CustomUser

# Register your models here.
from.models import *

admin.site.register(CustomUser)
admin.site.register(Code)
admin.site.register(Profile)
admin.site.register(Expense)