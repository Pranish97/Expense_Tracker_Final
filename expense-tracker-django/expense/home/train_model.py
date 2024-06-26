import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import os

# Load and preprocess data
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expense_dicti.pkl')

with open(file_path, 'rb') as f:
    category_dict = pickle.load(f)

df = pd.DataFrame(category_dict)
df['Category_encoded'] = df['Category'].astype('category').cat.codes
df['Subcategory_encoded'] = df['Subcategory'].astype('category').cat.codes

# Train Decision Tree Regressor
X = df[['Category_encoded', 'Subcategory_encoded']]
y = df['Amount']
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# Save the trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_tree_regressor.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(regressor, model_file)

print("Model trained and saved successfully.")
