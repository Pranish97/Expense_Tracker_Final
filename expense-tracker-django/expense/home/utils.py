import os
from twilio.rest import Client

from django.core.mail import send_mail
from django.conf import settings

# Your Account Sid and Auth Token from twilio.com/console
# and set the environment variables. See http://twil.io/secure
# account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
# auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

account_sid = "ACdfbbdb59eb416d5c47257fe2a86ef48d"
auth_token = "ff2486f28d32eba4bd0f278e95b88b1b"
client = Client(account_sid, auth_token)

def send_sms(user_code, phone_number):
    message = client.messages.create(
        body=f'Your verification code is {user_code}',
        from_='+13479604258',
        to=f'{phone_number}'
    )

    print(message.sid)

def send_email(code, recipient_email='pranishstha19@gmail.com'):
    subject = 'Expense Tracker - Verification Code' 
    message = f'Your verification code is {code}'
    from_email = settings.EMAIL_HOST_USER
    recipient_list = [recipient_email]

    try:
        send_mail(
            subject,
            message,
            from_email,
            recipient_list,
            fail_silently=False
        )

        print('Email sent to:', recipient_list)
    except Exception as e:
        print(f'Error sending email: {e}')