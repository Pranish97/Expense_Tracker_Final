from django.contrib.auth.models import auth
from .models import Expense, Profile
from django.shortcuts import redirect, render, get_object_or_404
from .models import *
from .forms import CreateUserForm, CustomUserChangeForm, LoginForm, AnomalyDetectionForm, CodeForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import auth
from django.contrib.auth import authenticate, login, logout, get_backends
from django.urls import reverse
from .utils import send_email
from django.utils import timezone
import json
import os
import pickle
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


@login_required(login_url="login")
def set_limit(request):
    allowed_categories = [
        'Transportation', 'Food', 'Investment', 'Family', 'Festivals', 
        'subscription', 'Life Insurance', 'Health', 'Beauty', 'Rent', 'Education'
    ]

    if request.method == 'POST':
        form = AnomalyDetectionForm(request.POST, categories=allowed_categories)
        if form.is_valid():
            category = form.cleaned_data['category']
            limit = form.cleaned_data['limit']
            
            profile = Profile.objects.get(user=request.user)
            profile.set_limit(category, limit)
            
            messages.success(request, f'Limit set for {category} successfully!')
            return redirect('home')
    else:
        form = AnomalyDetectionForm(categories=allowed_categories)
    
    return render(request, 'limit.html', {'form': form})


@login_required(login_url="login")
def home(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'expense_dicti.pkl')

    try:
        with open(file_path, 'rb') as f:
            category_dict = pickle.load(f)
    except FileNotFoundError:
        return render(request, 'file_not_found.html')

    allowed_categories = [
        'Transportation', 'Food', 'Investment', 'Family', 'Festivals', 
        'Subscription', 'Life Insurance', 'Health', 'Beauty', 'Rent', 'Education'
    ]

    df = pd.DataFrame(category_dict)
    df = df[df['Category'].isin(allowed_categories)]

    profile = Profile.objects.filter(user=request.user).first()
    expenses = Expense.objects.filter(user=request.user).order_by('date')

    category_limits = {category: profile.get_limit(category) for category in allowed_categories}

    # Aggregate cumulative total expenses by date
    cumulative_expenses = []
    cumulative_total = 0
    for expense in expenses.filter(expense_type='Negative').order_by('date'):
        cumulative_total += expense.amount
        cumulative_expenses.append({
            'date': expense.date.strftime('%Y-%m-%d'),
            'cumulative_total': cumulative_total
        })

    # Prepare data for Chart.js
    labels = [entry['date'] for entry in cumulative_expenses]
    data = [entry['cumulative_total'] for entry in cumulative_expenses]

    if request.method == 'POST':
        if 'add' in request.POST:
            category = request.POST.get('selected_category')
            amount_str = request.POST.get('amount')
            amount = float(amount_str)
            expense_type = request.POST.get('expense_type')

            subcategory = find_subcategory(df, category, amount)
            
            limit = profile.get_limit(category)
            total_expenses_for_category = sum(
                expense.amount for expense in expenses if expense.name == category and expense.expense_type == 'Negative'
            )
            
            if limit is not None and (total_expenses_for_category + amount > limit):
                messages.error(request, f'Cannot add expense. Total expenses for {category} exceed the limit of {limit}.')
            elif profile.balance >= amount or expense_type == 'Positive':
                # Anomaly Detection
                if expense_type == 'Negative':  # Only detect anomalies for expenses, not incomes
                    is_anomaly = detect_anomaly(category, amount, expenses)
                    if is_anomaly and not request.POST.get('confirm_anomaly'):
                        context = {
                            'profile': profile,
                            'expenses': expenses,
                            'categories': df['Category'].unique(),
                            'category_limits': category_limits,
                            'labels': json.dumps(labels),  
                            'expense_data': json.dumps(data),  
                            'is_anomaly': True,
                            'category': category,
                            'amount': amount,
                            'expense_type': expense_type,
                        }
                        return render(request, 'home.html', context)

                expense = Expense(
                    user=request.user,
                    name=category,
                    subcategory=subcategory,
                    amount=amount,
                    expense_type=expense_type,
                    date=timezone.now()  
                )
                expense.save()

                if expense_type == 'Positive':
                    profile.balance += amount
                else:
                    profile.expenses += amount
                    profile.balance -= amount

                profile.save()
                messages.success(request, 'Transaction Added Successfully!')
            else:
                messages.error(request, "Insufficient balance to make this expense.")
        elif 'edit' in request.POST:
            expense_id = request.POST.get('expense_id')
            expense = get_object_or_404(Expense, id=expense_id, user=request.user)

            context = {
                'profile': profile,
                'expenses': expenses,
                'expense': expense,
                'categories': df['Category'].unique(),
                'category_limits': category_limits,
                'labels': json.dumps(labels), 
                'expense_data': json.dumps(data),  
            }
            return render(request, 'home.html', context)
        elif 'delete' in request.POST:
            expense_id = request.POST.get('expense_id')
            expense = get_object_or_404(Expense, id=expense_id, user=request.user)

            if expense.expense_type == 'Positive':
                profile.balance -= expense.amount
            else:
                profile.expenses -= expense.amount
                profile.balance += expense.amount

            profile.save()
            messages.success(request, 'Transaction Deleted Successfully!')
            expense.delete()
        elif 'update' in request.POST:
            expense_id = request.POST.get('expense_id')
            category = request.POST.get('selected_category')
            amount_str = request.POST.get('amount')
            amount = float(amount_str)
            expense_type = request.POST.get('expense_type')

            expense = get_object_or_404(Expense, id=expense_id, user=request.user)

            if expense.expense_type == 'Positive':
                profile.balance -= expense.amount
            else:
                profile.expenses -= expense.amount
                profile.balance += expense.amount

            if expense_type == 'Positive':
                profile.balance += amount
            else:
                profile.expenses += amount
                profile.balance -= amount

            expense.name = category
            expense.subcategory = find_subcategory(df, category, amount)
            expense.amount = amount
            expense.expense_type = expense_type
            expense.save()

            profile.save()
            messages.info(request, 'Transaction Updated Successfully!')

        return redirect('home')

    context = {
        'profile': profile,
        'expenses': expenses,
        'categories': df['Category'].unique(),
        'category_limits': category_limits,
        'labels': json.dumps(labels),  
        'expense_data': json.dumps(data),  
        'is_anomaly': False,  
    }
    return render(request, 'home.html', context)

def find_subcategory(df, category, amount):
    # Prepare training data
    X_train = df[['Category', 'Amount', 'Subcategory']].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for feature in ['Category', 'Subcategory']:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        label_encoders[feature] = le
    
    # Train the model (using a simple K-nearest neighbors for demonstration)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train[['Category', 'Amount']], X_train['Subcategory'])
    
    # Prepare new data for prediction
    X_new = pd.DataFrame({
        'Category': [category],
        'Amount': [amount],
    })
    for feature in ['Category']:
        X_new[feature] = label_encoders[feature].transform(X_new[feature])
    
    # Predict subcategory
    predicted_subcategory = model.predict(X_new[['Category', 'Amount']])[0]
    
    return predicted_subcategory

def detect_anomaly(category, amount, expenses):
    # Create a DataFrame from the expenses for the specified category
    historical_data = pd.DataFrame(list(expenses.filter(name=category, expense_type='Negative').values('date', 'amount')))
    
    # Check if the 'date' column exists
    if 'date' not in historical_data.columns:
        print("Error: 'date' column is missing from historical_data")
        return False  # Or some other appropriate action
    
    # Convert 'date' column to datetime
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data['day_of_week'] = historical_data['date'].dt.dayofweek
    historical_data['average_amount'] = historical_data['amount'].rolling(window=7, min_periods=1).mean()

    # Prepare the features for the model
    X = historical_data[['amount', 'day_of_week', 'average_amount']]
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.1, n_estimators=100, max_samples='auto', random_state=42)
    model.fit(X)

    # Prepare new transaction data
    new_transaction_data = pd.DataFrame({
        'date': [timezone.now()],
        'amount': [amount]
    })
    new_transaction_data['day_of_week'] = new_transaction_data['date'].dt.dayofweek
    new_transaction_data['average_amount'] = historical_data['amount'].mean() 

    new_X = new_transaction_data[['amount', 'day_of_week', 'average_amount']]
    print("New transaction data:")
    print(new_X)

    is_anomaly = model.predict(new_X)
    print("Is anomaly:", is_anomaly)

    return is_anomaly[0] == -1


def user_register(request):
    form = CreateUserForm()  

    if request.method == "POST":  
        form = CreateUserForm(request.POST)  

        if form.is_valid():  
            form.save()  
            messages.success(request, 'User registered successfully!')  
            return redirect("login") 
        
    context = {'registerform': form}  

    return render(request, 'register.html', context=context) 

def user_login(request):

    form = LoginForm()

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                # auth.login(request, user)
  
                # return redirect("/")

                request.session['pk'] = user.pk
                return redirect('verify_otp');
    
    context = {'loginform':form}

    return render(request, 'login.html', context=context)

@login_required(login_url="login")
def user_logout(request):

    auth.logout(request)
    messages.success(request, 'User Logged Out Successfully!') 

    return redirect("login")


def recommend(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'expense_dicti.pkl')

    try:
        with open(file_path, 'rb') as f:
            category_dict = pickle.load(f)
    except FileNotFoundError:
        return render(request, 'file_not_found.html')

    categories = pd.DataFrame(category_dict)['Category'].unique()

    # Load the trained Decision Tree Regressor model
    model_path = os.path.join(current_dir, 'decision_tree_regressor.pkl')
    with open(model_path, 'rb') as model_file:
        regressor = pickle.load(model_file)

    if request.method == 'POST':
        df = pd.DataFrame(category_dict)

        selected_category_name = request.POST.get('selected_category')
        amount = float(request.POST.get('amount'))

        # Encode selected category
        df['Category_encoded'] = df['Category'].astype('category').cat.codes
        selected_category_encoded = df['Category_encoded'][df['Category'] == selected_category_name].iloc[0]

        # Prepare data for prediction
        df['Subcategory_encoded'] = df['Subcategory'].astype('category').cat.codes
        X_new = df[['Category_encoded', 'Subcategory_encoded']]
        predicted_amounts = regressor.predict(X_new)

        # Add predictions to dataframe
        df['Predicted_Amount'] = predicted_amounts

        # Filter subcategories based on the selected category and predicted amounts
        subcategories = df[(df['Category'] == selected_category_name) & (df['Predicted_Amount'] <= amount)]
        recommended_subcategories = []
        remaining_amount = amount

        for index, row in subcategories.iterrows():
            if remaining_amount >= row['Predicted_Amount']:
                recommended_subcategories.append((row['Subcategory'], row['Predicted_Amount'], row['Note']))
                remaining_amount -= row['Predicted_Amount']

        return render(request, 'recommend.html', {'selected_category_name': selected_category_name,
                                                   'amount': amount,
                                                   'recommended_subcategories': recommended_subcategories,
                                                   'categories': categories})

    else:
        return render(request, 'recommend.html', {'categories': categories})

@login_required(login_url="login")
def update_profile(request):
    user = request.user
    form = CustomUserChangeForm(instance=user)

    if request.method == 'POST':
        form = CustomUserChangeForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.info(request, 'User Updated Successfully!')  
            return redirect('/')  

    context = {'form': form}
    return render(request, 'profile.html', context)

@login_required(login_url="login")
def budget_profile(request):
    profile = Profile.objects.filter(user=request.user).first()

    context = {'profile': profile}

    return render(request, 'budget.html', context)

@login_required(login_url="login")
def update_budget_profile(request):
    if request.method == 'POST':
        profile = Profile.objects.filter(user=request.user).first()

        profile.income = request.POST.get('income')
        profile.expenses = request.POST.get('expenses')
        profile.balance = request.POST.get('balance')
        profile.save()
        messages.success(request, 'Budget Updated Successfully!')  

        return redirect('/')
    
def verify_otp(request):
    form = CodeForm(request.POST or None)
    pk = request.session.get('pk')

    if pk:
        user = CustomUser.objects.get(pk=pk)
        code = user.code
        code_user = f"{user.username}: {code}"

        if not request.POST:
            print(code_user)
            # send sms
            # send_sms(code_user, user.phone_number)

            # send email
            send_email(code, 'pranishjayana2@gmail.com')
        
        if form.is_valid():
            num = form.cleaned_data.get('number')

            if str(code) == num:
                code.save()
                # Set the backend attribute on the user
                backend = get_backends()[0]
                user.backend = f"{backend.__module__}.{backend.__class__.__name__}"
                login(request, user)
                return redirect("/")
            else:
                return redirect('login')
    
    return render(request, 'two_factor_auth.html', {'form': form})

def forecast_expenses(request):
    # Example expenses data
    expenses = Expense.objects.filter(user=request.user).order_by('date')
    data = pd.DataFrame(list(expenses.values('date', 'amount')))
    
    if data.empty:
        return render(request, 'forecast.html', {'forecast_data': '[]'})  # Return empty array if no data
    
    # Prepare data for Prophet
    data.rename(columns={'date': 'ds', 'amount': 'y'}, inplace=True)
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)  # Remove timezone information

    # Train Prophet model
    model = Prophet()
    model.fit(data)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Select relevant columns and convert Timestamp to datetime
    forecast['ds'] = forecast['ds'].dt.to_pydatetime()

    # Prepare forecast data as a list of dictionaries
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')

    # Serialize forecast_data to JSON string
    forecast_data_json = json.dumps(forecast_data, default=str)  # Use default=str to serialize datetimes

    return render(request, 'forecast.html', {'forecast_data': forecast_data_json})