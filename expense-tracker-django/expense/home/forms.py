from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm
from django.contrib.auth.models import User
from django.core.validators import RegexValidator

from django import forms

from home.models import CustomUser

# from django.forms.widgets import PasswordInput, TextInput

# - Create/Register a user (Model Form)
class CreateUserForm(UserCreationForm):
    username = forms.CharField(required=False, widget=forms.TextInput(attrs={
        "class": "input",
        "type": "text",
        "placeholder": "enter username"
    }))

    email = forms.CharField(required=False, widget=forms.TextInput(attrs={
        "class": "input",
        "type": "text",
        "placeholder": "enter email-id"
    }))

    password1 = forms.CharField(required=False, widget=forms.TextInput(attrs={
        "class": "input",
        "type": "password",
        "placeholder": "enter password"
    }))

    password2 = forms.CharField(required=False, widget=forms.TextInput(attrs={
        "class": "input",
        "type": "password",
        "placeholder": "re-enter password"
    }))

    phone_number = forms.CharField(required=False, widget=forms.TextInput(attrs={
        "class": "input",
        "type": "tel",
        "placeholder": "Enter phone number"
    }))

    # email_validator = RegexValidator(
    #     regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    #     message='Please enter a valid email address.'
    # )

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'phone_number']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if not username:
            raise forms.ValidationError("Username cannot be empty.")
        return username
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not email:
            raise forms.ValidationError("Email cannot be empty.")
        return email

    # def clean_email(self):
    #     email = self.cleaned_data.get('email')
    #     if not email:
    #         raise forms.ValidationError("Email cannot be empty.")
    #     self.email_validator(email)  # Validate email format using regex
    #     return email
    
    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        if not password1:
            raise forms.ValidationError("Password cannot be empty.")
        return password1
    
    def clean_password2(self):
        password2 = self.cleaned_data.get('password2')
        if not password2:
            raise forms.ValidationError("Confirm password cannot be empty.")
        return password2
    
    def clean_phone_number(self):
        phone_number = self.cleaned_data.get('phone_number')
        if not phone_number:
            raise forms.ValidationError("Phone number cannot be empty.")
        return phone_number

# - Authenticate a user (Model Form)
class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)

    username = forms.CharField(widget=forms.TextInput(attrs={
        "class": "input",
        "type": "text",
        "placeholder": "enter username"
    }))

    password = forms.CharField(widget=forms.TextInput(attrs={
        "class": "input",
        "type": "password",
        "placeholder": "enter password"
    }))

class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'phone_number')

class CodeForm(forms.ModelForm):
    number = forms.CharField(required=False, label='Code', help_text='Enter SMS verification code')

    class Meta:
        model = CustomUser
        fields = ('number',)


class AnomalyDetectionForm(forms.Form):
    category = forms.ChoiceField(choices=[])  
    limit = forms.FloatField()

    def __init__(self, *args, **kwargs):
        categories = kwargs.pop('categories', [])  
        super().__init__(*args, **kwargs)
        self.fields['category'].choices = [(category, category) for category in categories]