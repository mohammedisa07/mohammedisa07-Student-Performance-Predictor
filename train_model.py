import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Step 1: Generate synthetic data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)

    data = {
        'study_hours_per_week': np.random.normal(15, 5, n_samples).clip(0, 40),
        'attendance_rate': np.random.normal(85, 10, n_samples).clip(50, 100),
        'previous_grade': np.random.normal(75, 12, n_samples).clip(40, 100),
        'family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                             n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'extracurricular_activities': np.random.randint(0, 5, n_samples),
        'internet_access': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'tutoring': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(4, 12),
        'stress_level': np.random.randint(1, 11, n_samples),
        'motivation_level': np.random.randint(1, 11, n_samples)
    }

    df = pd.DataFrame(data)

    # Target variable
    final_grade = (
        df['study_hours_per_week'] * 1.2 +
        df['attendance_rate'] * 0.3 +
        df['previous_grade'] * 0.4 +
        df['extracurricular_activities'] * 2 +
        df['internet_access'] * 5 +
        df['tutoring'] * 8 +
        df['sleep_hours'] * 2 +
        df['motivation_level'] * 3 -
        df['stress_level'] * 1.5 +
        np.random.normal(0, 10, n_samples)
    ).clip(0, 100)

    df['final_grade'] = final_grade
    return df

# Step 2: Preprocess and train model
df = generate_sample_data()

# Label encode categorical features
label_encoders = {}
for col in ['family_income', 'parent_education']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('final_grade', axis=1)
y = df['final_grade']

feature_names = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
feature_selector = SelectKBest(score_func=f_regression, k=8)
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = feature_selector.transform(X_test_scaled)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Step 3: Save everything
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(feature_selector, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("✅ Model and preprocessing components saved.")
