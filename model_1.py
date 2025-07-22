# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix
import joblib
import ast  # For safely evaluating strings as lists
import matplotlib.pyplot as plt
import seaborn as sns

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
# Check column names in the merged dataset
print("Worker_Allocation Columns:", worker_allocation.columns)
print("Worker_Task_Allocation Columns:", worker_task_allocation.columns)
print("Merged Data Columns:", merged_data.columns)

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

# Step 2.3: Verify Numerical Columns
# Initialize the scaler object
scaler = StandardScaler()

# Check if 'Workload_for_Shift' exists in the DataFrame
if 'Workload_for_Shift' not in merged_data.columns:
    print("Warning: 'Workload_for_Shift' column not found in the DataFrame. Adding it with default values.")
    merged_data['Workload_for_Shift'] = 0  # Add the column with default value 0

# Define numerical columns
numerical_cols = ['Years_of_Experience', 'Workload_for_Shift']

# Normalize numerical variables
merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])

# Step 3: Split the Dataset
# Features (X) and target (y) for regression (Performance Rating Prediction)
X_reg = merged_data.drop(columns=['Performance_Rating', 'Current_Task_Assigned'])
y_reg = merged_data['Performance_Rating']

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
classifier.fit(X_clf_train, y_clf_train)

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

# Confusion Matrix
cm = confusion_matrix(y_clf_test_decoded, y_clf_pred_decoded)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder_y.classes_, yticklabels=label_encoder_y.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Task Recommendation')
plt.show()

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

# Step 6: Deployment Example
# Load the models and preprocessing objects
regressor = joblib.load(r'models/performance_rating_predictor.pkl')
classifier = joblib.load(r'models/task_recommendation_classifier.pkl')
label_encoder_y = joblib.load(r'models/label_encoder_y.pkl')
scaler = joblib.load(r'models/scaler.pkl')
label_encoder_skill_level = joblib.load(r'models/label_encoder_skill_level.pkl')

# Example: Predict performance rating for a worker
# Create a sample worker data (replace with actual data)
worker_data = pd.DataFrame({
    'Worker_ID': [1],
    'Skill_Category': ['Welding'],
    'Skill_Level': ['High'],  # Ensure this value is in the training data
    'Years_of_Experience': [5],
    'Current_Task_Assigned': ['Cell Assembly'],
    'Shift_Availability': ['Morning'],
    'Certifications': ['Certified Welder'],
    'Task_Priority': ['Critical'],
    'Workload_for_Shift': [10]  # Include this column even if it was missing earlier
})

# Add skill columns to worker_data (initialize with 0)
for skill in all_task_skills:
    worker_data[skill] = 0

# Set the relevant skills for the worker (e.g., 'Welding')
worker_data['Welding'] = 1

# Preprocess the worker data
# Label encode categorical columns
worker_data['Skill_Level'] = label_encoder_skill_level.transform(worker_data['Skill_Level'])
worker_data['Task_Priority'] = LabelEncoder().fit_transform(worker_data['Task_Priority'])

# One-hot encode categorical columns
worker_data = pd.get_dummies(worker_data, columns=categorical_cols, drop_first=True)

# Normalize numerical columns
worker_data[numerical_cols] = scaler.transform(worker_data[numerical_cols])

# Ensure all columns in worker_data match the training data
# Add missing columns (if any) and set their values to 0
missing_cols = set(X_reg_train.columns) - set(worker_data.columns)
for col in missing_cols:
    worker_data[col] = 0

# Reorder columns to match the training data
worker_data = worker_data[X_reg_train.columns]

# Predict performance rating
performance_rating = regressor.predict(worker_data)
print(f"Predicted Performance Rating: {performance_rating}")

# Recommend a task for the worker
recommended_task_encoded = classifier.predict(worker_data)
recommended_task = label_encoder_y.inverse_transform(recommended_task_encoded)
print(f"Recommended Task: {recommended_task}")