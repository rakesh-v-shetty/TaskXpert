import pandas as pd
import random

# List of possible workers' skills, tasks, shift availability, and certifications
skills = ['Welding', 'BMS Integration', 'Mechanical Assembly', 'Electrical Wiring', 'Quality Inspection', 'Automation', 'Robotics', 'Soldering', 'Packaging']
tasks = ['Cell Assembly', 'Module Assembly', 'Pack Assembly', 'Final Testing']
shift_availability = ['Morning', 'Evening', 'Night', 'Flexible']
certifications = ['Certified Welder', 'Certified BMS Engineer', 'Mechanical Technician', 'Electrical Engineer', 'Quality Control Certified', 'None']

# Skills required for each station
task_skills = {
    'Cell Assembly': ['Welding', 'Electrical Wiring', 'Mechanical Assembly'],
    'Module Assembly': ['BMS Integration', 'Electrical Wiring', 'Mechanical Assembly'],
    'Pack Assembly': ['Welding', 'Mechanical Assembly', 'Robotics'],
    'Final Testing': ['Quality Inspection', 'Electrical Wiring', 'Automation']
}

# Workload per shift (number of tasks expected to be completed per shift)
shift_workload = {
    'Morning': 15,
    'Evening': 12,
    'Night': 10,
    'Flexible': 14
}

# Priority of stations (Critical or Secondary)
station_priority = {
    'Cell Assembly': 'Critical',
    'Module Assembly': 'Critical',
    'Pack Assembly': 'Secondary',
    'Final Testing': 'Critical'
}

# Function to generate realistic data for workers with task-related information
def generate_worker_task_data(worker_id):
    skill = random.choice(skills)
    skill_level = random.choice(['Low', 'Medium', 'High'])
    years_experience = random.randint(1, 8)
    task = random.choice(list(tasks))
    shift = random.choice(list(shift_availability))
    performance_rating = round(random.uniform(3.0, 5.0), 1)
    certification = random.choice(certifications)
    task_skills_required = task_skills[task]
    task_priority = station_priority[task]
    workload_for_shift = shift_workload[shift]
    
    return {
        'Worker ID': worker_id,
        'Skill Category': skill,
        'Skill Level': skill_level,
        'Years of Experience': years_experience,
        'Current Task Assigned': task,
        'Shift Availability': shift,
        'Performance Rating': performance_rating,
        'Certifications': certification,
        'Task Skills Required': task_skills_required,
        'Task Priority': task_priority,
        'Workload for Shift': workload_for_shift
    }

# Generate 1000 worker task entries
worker_task_data = [generate_worker_task_data(i) for i in range(1, 1001)]

# Create the enhanced DataFrame
worker_task_df = pd.DataFrame(worker_task_data)

# Saving the dataset to a CSV file
file_path = 'worker_task_allocation_dataset_1000.csv'
worker_task_df.to_csv(file_path, index=False)

file_path  # Returning the path for download
