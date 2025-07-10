
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load data
df = pd.read_csv('../data/sample_data.csv')

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
df['days_since_last_purchase'] = (df['timestamp'].max() - df['last_purchase_date']).dt.days
df['purchase_made'] = df['purchase_amount'].apply(lambda x: 1 if x > 0 else 0)

df = pd.get_dummies(df, columns=['product_category', 'gender'], drop_first=True)

df = df.drop(columns=['user_id', 'timestamp', 'last_purchase_date', 'purchase_amount'])

# Scale
scaler = StandardScaler()
df[['time_spent', 'price', 'clicks', 'days_since_last_purchase']] = scaler.fit_transform(
    df[['time_spent', 'price', 'clicks', 'days_since_last_purchase']])

X = df.drop('purchase_made', axis=1)
y = df['purchase_made']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
