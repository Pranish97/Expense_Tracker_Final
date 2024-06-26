from django.db.models.signals import post_save
from django.dispatch import receiver  
from home.models import CustomUser, Code

@receiver(post_save, sender=CustomUser)
def post_save_generate_code(sender, instance, created, **kwargs):
    if created:
        Code.objects.create(user=instance)