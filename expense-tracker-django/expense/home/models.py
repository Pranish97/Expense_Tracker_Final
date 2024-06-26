import random
from django.db import models
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

# Create your models here.

TYPE = (
    ('Positive', 'Positive'),
    ('Negative', 'Negativ')
)

class CustomUser(AbstractUser):
    phone_number = models.CharField(max_length=15)

class Code(models.Model):
    number = models.CharField(max_length=5, blank=True)
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)

    def __str__(self):
        return self.number
    
    def save(self, *args, **kwargs):
        number_list = [x for x in range(10)]
        code_items = []

        for i in range(5):
            num = random.choice(number_list)
            code_items.append(str(num))

        code_string = "".join(str(item) for item in code_items)
        self.number = code_string
        super().save(*args, **kwargs)

class Profile(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    income = models.FloatField()
    expenses = models.FloatField(default=0)
    balance = models.FloatField(blank=True, null=True)
    anomaly_limits = models.JSONField(default=dict)

    def set_limit(self, category, limit):
        self.anomaly_limits[category] = limit
        self.save()

    def get_limit(self, category):
        return self.anomaly_limits.get(category, None)

class Expense(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    subcategory = models.CharField(max_length=255, blank=True, null=True)
    name = models.CharField(max_length=100)
    amount = models.FloatField()
    expense_type = models.CharField(max_length=100, choices=TYPE)
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name

@receiver(post_save, sender=CustomUser)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance, income=0, expenses=0, balance=0)