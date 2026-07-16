import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    
    data = {
        'category_id': np.random.randint(0, 10, n_samples),
        'priority_level': np.random.randint(0, 3, n_samples), # 0:Low, 1:Med, 2:High
        'zone_tier': np.random.randint(1, 5, n_samples),     # 1:Metro to 4:Rural
        'officer_load': np.random.randint(0, 21, n_samples),
        'has_media': np.random.randint(0, 2, n_samples),
        'is_gps_verified': np.random.randint(0, 2, n_samples),
        'cluster_size': np.random.randint(1, 101, n_samples)
    }
    
    df = pd.DataFrame(data)

    # Base days logic (The "Target")
    # Water/Elec (0,2) are fast, Roads/Corruption (1,8) are slow
    base_days = {0:2, 1:7, 2:1, 3:4, 4:3, 5:5, 6:8, 7:10, 8:15, 9:5}
    df['res_days'] = df['category_id'].map(base_days).astype(float)

    # Modifiers
    df['res_days'] *= np.where(df['priority_level'] == 2, 0.6, 1.0) # High priority is 40% faster
    df['res_days'] *= np.where(df['priority_level'] == 0, 1.4, 1.0) # Low priority is 40% slower
    df['res_days'] += (df['officer_load'] * 0.1)                   # Each task adds 0.1 days
    df['res_days'] -= (df['is_gps_verified'] * 0.5)               # Verified data is faster to process
    df['res_days'] -= np.where(df['cluster_size'] > 50, 1.5, 0)   # Mass issues get political speedup
    
    # Add some "Real World" noise
    noise = np.random.normal(0, 0.5, n_samples)
    df['res_days'] = (df['res_days'] + noise).clip(0.5, 21) # Min 12 hours, Max 21 days
    
    return df

# 1. Generate and Split
df = generate_synthetic_data()
X = df.drop('res_days', axis=1)
y = df['res_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train XGBoost
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

# 3. Evaluate
predictions = model.predict(X_test)
print(f"Model Error: {mean_absolute_error(y_test, predictions):.2f} days")

# 4. Save
with open("../weights/sla_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SLA Model saved to weights/")