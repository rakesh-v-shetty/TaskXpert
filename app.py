from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models and preprocessing objects
regressor = joblib.load('models/performance_rating_predictor.pkl')
classifier = joblib.load('models/task_recommendation_classifier.pkl')
label_encoder_y = joblib.load('models/label_encoder_y.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder_skill_level = joblib.load('models/label_encoder_skill_level.pkl')

X_reg_columns = joblib.load('models/X_reg_columns.pkl')

all_task_skills = [...]  # Replace with actual skills

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        skill_category = request.form['skill_category']
        skill_level = request.form['skill_level']
        years_of_experience = float(request.form['years_of_experience'])
        shift_availability = request.form['shift_availability']
        certifications = request.form['certifications']
        current_task_assigned = request.form['current_task_assigned']

        task_priority_mapping = {
            'Cell Assembly': 'Critical',
            'Module Assembly': 'Critical',
            'Pack Assembly': 'Secondary',
            'Final Testing': 'Critical'
        }
        task_priority = task_priority_mapping.get(current_task_assigned, 'Secondary')

        worker_data = {
            'Skill_Category': skill_category,
            'Skill_Level': skill_level,
            'Years_of_Experience': years_of_experience,
            'Current_Task_Assigned': current_task_assigned,
            'Shift_Availability': shift_availability,
            'Certifications': certifications,
            'Task_Priority': task_priority
        }

        worker_df = pd.DataFrame([worker_data])

        for skill in all_task_skills:
            worker_df[skill] = 0
        worker_df[skill_category] = 1

        worker_df['Skill_Level'] = label_encoder_skill_level.transform(worker_df['Skill_Level'])
        worker_df['Task_Priority'] = pd.factorize(worker_df['Task_Priority'])[0]

        categorical_cols = ['Skill_Category', 'Shift_Availability', 'Certifications', 'Current_Task_Assigned']
        worker_df = pd.get_dummies(worker_df, columns=categorical_cols, drop_first=True)

        worker_df[['Years_of_Experience']] = scaler.transform(worker_df[['Years_of_Experience']])

        missing_cols = set(X_reg_columns) - set(worker_df.columns)
        for col in missing_cols:
            worker_df[col] = 0
        worker_df = worker_df[X_reg_columns]

        performance_rating = regressor.predict(worker_df)[0]
        worker_df_clf = worker_df.drop(columns=['Performance_Rating', 'Current_Task_Assigned'], errors='ignore')
        task_prediction = classifier.predict(worker_df_clf)[0]
        recommended_task = label_encoder_y.inverse_transform([task_prediction])[0]

        return render_template('model.html', performance_rating=performance_rating, recommended_task=recommended_task)

    return render_template('model.html', performance_rating=None, recommended_task=None)

if __name__ == '__main__':
    app.run(debug=True)
