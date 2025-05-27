from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

class StudentPerformanceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = None
        self.is_trained = False

    def generate_sample_data(self, n_samples=1000):
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

    def train_model(self):
        df = self.generate_sample_data()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop('final_grade', axis=1)
        y = df['final_grade']
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.feature_selector = SelectKBest(score_func=f_regression, k=8)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_selected, y_train)

        y_pred = self.model.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        self.is_trained = True
        return {'rmse': rmse, 'r2': r2}

    def predict(self, student_data):
        if not self.is_trained:
            self.train_model()

        df = pd.DataFrame([student_data])
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        df = df[self.feature_names]
        df_scaled = self.scaler.transform(df)
        df_selected = self.feature_selector.transform(df_scaled)

        prediction = self.model.predict(df_selected)[0]
        return max(0, min(100, prediction))

    def get_feature_importance(self):
        if not self.is_trained:
            return {}

        selected_features = np.array(self.feature_names)[self.feature_selector.get_support()]
        importance = self.model.feature_importances_
        return dict(zip(selected_features, importance))


model = StudentPerformanceModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debug

        student_data = {
            'study_hours_per_week': float(data['study_hours']),
            'attendance_rate': float(data['attendance']),
            'previous_grade': float(data['previous_grade']),
            'family_income': data['family_income'],
            'parent_education': data['parent_education'],
            'extracurricular_activities': int(data['extracurricular']),
            'internet_access': 1 if data['internet_access'] == 'Yes' else 0,
            'tutoring': 1 if data['tutoring'] == 'Yes' else 0,
            'sleep_hours': float(data['sleep_hours']),
            'stress_level': int(data['stress_level']),
            'motivation_level': int(data['motivation_level'])
        }

        prediction = model.predict(student_data)

        if prediction >= 90:
            category = "Excellent"
            color = "#10B981"
        elif prediction >= 80:
            category = "Good"
            color = "#3B82F6"
        elif prediction >= 70:
            category = "Average"
            color = "#F59E0B"
        elif prediction >= 60:
            category = "Below Average"
            color = "#EF4444"
        else:
            category = "Poor"
            color = "#DC2626"

        recommendations = generate_recommendations(student_data, prediction)

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'category': category,
            'color': color,
            'recommendations': recommendations
        })

    except Exception as e:
        print("Prediction error:", str(e))  # Debug log
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analytics')
def analytics():
    try:
        if not model.is_trained:
            model.train_model()

        feature_importance = model.get_feature_importance()
        return jsonify({'success': True, 'feature_importance': feature_importance})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_recommendations(student_data, prediction):
    recommendations = []

    if student_data['study_hours_per_week'] < 15:
        recommendations.append("Increase study hours to at least 15–20 hours per week.")

    if student_data['attendance_rate'] < 85:
        recommendations.append("Improve your attendance for better understanding and retention.")

    if student_data['sleep_hours'] < 7:
        recommendations.append("Get 7–9 hours of sleep for optimal brain performance.")

    if student_data['stress_level'] > 7:
        recommendations.append("Manage stress through relaxation, breaks, or counseling.")

    if student_data['motivation_level'] < 6:
        recommendations.append("Set clear goals and reward yourself for small wins.")

    if student_data['tutoring'] == 0 and prediction < 75:
        recommendations.append("Consider tutoring to reinforce weak subjects.")

    if student_data['extracurricular_activities'] == 0:
        recommendations.append("Engage in at least one extracurricular activity for balance.")

    if not recommendations:
        recommendations.append("Excellent work! Stay consistent with your habits.")

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
