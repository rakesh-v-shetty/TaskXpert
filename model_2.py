# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import ast  # For safely evaluating strings as lists

# Step 1: Load and Merge Datasets
# Load datasets
worker_allocation = pd.read_csv(r'dataset\worker_allocation_dataset_1000\worker_allocation_dataset_1000.csv')
worker_task_allocation = pd.read_csv(r'dataset\worker_task_allocation_dataset_1000\worker_task_allocation_dataset_1000.csv')

# Standardize column names
worker_allocation.columns = worker_allocation.columns.str.strip().str.replace(' ', '_')
worker_task_allocation.columns = worker_task_allocation.columns.str.strip().str.replace(' ', '_')

# Merge datasets on Worker ID
merged_data = pd.merge(worker_allocation, worker_task_allocation, on='Worker_ID', how='inner', suffixes=('_left', '_right'))

# Step 2: Data Preprocessing
# Resolve overlapping columns by keeping only one version (e.g., from the left dataset)
for col in worker_allocation.columns:
    if col + '_left' in merged_data.columns:
        merged_data[col] = merged_data[col + '_left']
    elif col + '_right' in merged_data.columns:
        merged_data[col] = merged_data[col + '_right']

# Drop the suffixed columns
merged_data = merged_data.drop(columns=[col for col in merged_data.columns if col.endswith('_left') or col.endswith('_right')])

# Step 2.1: Preprocess 'Task_Skills_Required' Column
# Convert the 'Task_Skills_Required' column from string to list
merged_data['Task_Skills_Required'] = merged_data['Task_Skills_Required'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Flatten the list of skills and get all unique skills
all_task_skills = list(set(skill for sublist in merged_data['Task_Skills_Required'] for skill in sublist))

# Create a binary matrix for skills (one-hot encoding for multiple skills)
skill_matrix = merged_data['Task_Skills_Required'].apply(lambda x: pd.Series([1 if skill in x else 0 for skill in all_task_skills], index=all_task_skills))

# Concatenate the skill matrix with the original data
merged_data = pd.concat([merged_data, skill_matrix], axis=1)

# Drop the original 'Task_Skills_Required' column
merged_data = merged_data.drop(columns=['Task_Skills_Required'])

# Step 2.2: Encode Categorical Columns
# Define all possible values for 'Skill_Level'
all_skill_levels = ['Low', 'Medium', 'High']

# Fit LabelEncoder on all possible values
label_encoder_skill_level = LabelEncoder()
label_encoder_skill_level.fit(all_skill_levels)

# Transform the 'Skill_Level' column using the fitted LabelEncoder
merged_data['Skill_Level'] = label_encoder_skill_level.transform(merged_data['Skill_Level'])

# Update categorical columns for one-hot encoding
categorical_cols = ['Skill_Category', 'Shift_Availability', 'Certifications']

# One-hot encode categorical variables (excluding 'Current_Task_Assigned')
merged_data = pd.get_dummies(merged_data, columns=categorical_cols, drop_first=True)

# Label encode Task Priority (Critical = 1, Secondary = 0)
merged_data['Task_Priority'] = LabelEncoder().fit_transform(merged_data['Task_Priority'])

# Normalize numerical variables (removing 'Workload_for_Shift' from here)
numerical_cols = ['Years_of_Experience']
scaler = StandardScaler()
merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])

# Step 3: Split the Dataset
# Features (X) and target (y) for regression (Performance Rating Prediction)
X_reg = merged_data.drop(columns=['Performance_Rating', 'Current_Task_Assigned'])
y_reg = merged_data['Performance_Rating']

# After defining X_reg
X_reg_columns = X_reg.columns.tolist()
joblib.dump(X_reg_columns, r'models/X_reg_columns.pkl')

# Features (X) and target (y) for classification (Task Recommendation)
X_clf = merged_data.drop(columns=['Current_Task_Assigned', 'Performance_Rating'])
y_clf = merged_data['Current_Task_Assigned']

# Encode the target variable for classification (y_clf)
label_encoder_y = LabelEncoder()
y_clf_encoded = label_encoder_y.fit_transform(y_clf)

# Split data into training and testing sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf_encoded, test_size=0.2, random_state=42)

# Step 4: Model Training
# 4.1 Performance Rating Prediction (Regression)
# Initialize and train the regressor
regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
regressor.fit(X_reg_train, y_reg_train)

# Predict on test data
y_reg_pred = regressor.predict(X_reg_test)

# Evaluate the model using RMSE
mse = mean_squared_error(y_reg_test, y_reg_pred)  # Calculate MSE
rmse = mse ** 0.5  # Manually calculate RMSE
print(f"RMSE for Performance Rating Prediction: {rmse}")

# 4.2 Task Recommendation (Classification)
# Initialize and train the classifier
classifier = XGBClassifier(objective='multi:softprob', num_class=len(label_encoder_y.classes_), n_estimators=100, random_state=42)

# Handle class imbalance with scale_pos_weight (using class weights)
class_weights = {i: len(y_clf_train) / (len(label_encoder_y.classes_) * sum(y_clf_train == i)) for i in range(len(label_encoder_y.classes_))}
classifier.fit(X_clf_train, y_clf_train, sample_weight=[class_weights[i] for i in y_clf_train])

# Predict on test data
y_clf_pred = classifier.predict(X_clf_test)

# Decode the predicted labels back to original task names
y_clf_pred_decoded = label_encoder_y.inverse_transform(y_clf_pred)
y_clf_test_decoded = label_encoder_y.inverse_transform(y_clf_test)

# Evaluate the model using accuracy and F1-score
accuracy = accuracy_score(y_clf_test_decoded, y_clf_pred_decoded)
f1 = f1_score(y_clf_test_decoded, y_clf_pred_decoded, average='weighted')
print(f"Accuracy for Task Recommendation: {accuracy}")
print(f"F1-Score for Task Recommendation: {f1}")

# Show confusion matrix and classification report for deeper insights
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test_decoded, y_clf_pred_decoded))
print("Classification Report:")
print(classification_report(y_clf_test_decoded, y_clf_pred_decoded))

# Step 5: Save the Models and Preprocessing Objects
# Save the regression model
joblib.dump(regressor, r'models/performance_rating_predictor.pkl')

# Save the classification model
joblib.dump(classifier, r'models/task_recommendation_classifier.pkl')

# Save the label encoder for the target variable
joblib.dump(label_encoder_y, r'models/label_encoder_y.pkl')

# Save the scaler for numerical columns
joblib.dump(scaler, r'models/scaler.pkl')

# Save the label encoder for categorical columns
joblib.dump(label_encoder_skill_level, r'models/label_encoder_skill_level.pkl')

# Step 6: Deployment Example with User Input
# Load the models and preprocessing objects
regressor = joblib.load(r'models/performance_rating_predictor.pkl')
classifier = joblib.load(r'models/task_recommendation_classifier.pkl')
label_encoder_y = joblib.load(r'models/label_encoder_y.pkl')
scaler = joblib.load(r'models/scaler.pkl')
label_encoder_skill_level = joblib.load(r'models/label_encoder_skill_level.pkl')

# Function to predict worker's performance and recommend a task based on user input
def predict_worker_performance():
    # Get user input
    print("Please provide the following information:")
    skill_category = input("Enter Skill Category (e.g., Welding, Assembly): ")
    skill_level = input("Enter Skill Level (Low/Medium/High): ")
    years_of_experience = float(input("Enter Years of Experience: "))
    shift_availability = input("Enter Shift Availability (Morning/Night/Evening): ")
    certifications = input("Enter Certifications (if any, else enter 'None'): ")
    current_task_assigned = input("Enter Current Task Assigned (Cell Assembly/Module Assembly/Pack Assembly/Final Testing): ")

    # Derive 'Task_Priority' from 'Current_Task_Assigned'
    task_priority_mapping = {
        'Cell Assembly': 'Critical',
        'Module Assembly': 'Critical',
        'Pack Assembly': 'Secondary',
        'Final Testing': 'Critical'
    }
    task_priority = task_priority_mapping.get(current_task_assigned, 'Secondary')  # Default to 'Secondary' if task is unknown

    # Prepare worker data for prediction
    worker_data = {
        'Skill_Category': skill_category,
        'Skill_Level': skill_level,
        'Years_of_Experience': years_of_experience,
        'Current_Task_Assigned': current_task_assigned,
        'Shift_Availability': shift_availability,
        'Certifications': certifications,
        'Task_Priority': task_priority
    }

    # Create DataFrame from the user input
    worker_df = pd.DataFrame([worker_data])

    # Initialize skill columns (all set to 0 initially)
    for skill in all_task_skills:
        worker_df[skill] = 0

    # Set the relevant skill for the worker (e.g., 'Welding' in this example)
    worker_df[skill_category] = 1

    # Preprocess the worker data:
    # 1. Label encode the 'Skill_Level' and 'Task_Priority' columns
    worker_df['Skill_Level'] = label_encoder_skill_level.transform(worker_df['Skill_Level'])
    worker_df['Task_Priority'] = LabelEncoder().fit_transform(worker_df['Task_Priority'])

    # 2. One-hot encode categorical columns
    categorical_cols = ['Skill_Category', 'Shift_Availability', 'Certifications', 'Current_Task_Assigned']
    worker_df = pd.get_dummies(worker_df, columns=categorical_cols, drop_first=True)

    # 3. Normalize numerical columns
    worker_df[['Years_of_Experience']] = scaler.transform(worker_df[['Years_of_Experience']])

    # Ensure all necessary columns are present and add missing columns with default value 0
    missing_cols = set(X_reg.columns) - set(worker_df.columns)
    for col in missing_cols:
        worker_df[col] = 0

    # Reorder the columns to match the training data columns
    worker_df = worker_df[X_reg.columns]

    # Step 1: Predict performance rating
    performance_rating = regressor.predict(worker_df)[0]
    print(f"Predicted Performance Rating: {performance_rating}")

    # Step 2: Recommend a task based on the current task and worker's skills
    worker_df_clf = worker_df.drop(columns=['Performance_Rating', 'Current_Task_Assigned'], errors='ignore')
    
    task_prediction = classifier.predict(worker_df_clf)[0]
    recommended_task = label_encoder_y.inverse_transform([task_prediction])[0]
    print(f"Recommended Task: {recommended_task}")

# Step 7: Test the deployment function
predict_worker_performance()
