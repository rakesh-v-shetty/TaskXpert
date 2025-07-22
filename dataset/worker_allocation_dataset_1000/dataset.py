import pandas as pd
import random

# List of possible workers' skills, tasks, shift availability, and certifications
skills = ['Welding', 'BMS Integration', 'Mechanical Assembly', 'Electrical Wiring', 'Quality Inspection', 'Automation', 'Robotics', 'Soldering', 'Packaging']
tasks = ['Cell Assembly', 'Module Assembly', 'Pack Assembly', 'Final Testing']
shift_availability = ['Morning', 'Evening', 'Night', 'Flexible']
certifications = ['Certified Welder', 'Certified BMS Engineer', 'Mechanical Technician', 'Electrical Engineer', 'Quality Control Certified', 'None']

# Function to generate realistic data for workers
def generate_worker_data(worker_id):
    skill = random.choice(skills)
    skill_level = random.choice(['Low', 'Medium', 'High'])
    years_experience = random.randint(1, 8)
    task = random.choice(tasks)
    shift = random.choice(shift_availability)
    performance_rating = round(random.uniform(3.0, 5.0), 1)
    certification = random.choice(certifications)
    
    return {
        'Worker ID': worker_id,
        'Skill Category': skill,
        'Skill Level': skill_level,
        'Years of Experience': years_experience,
        'Current Task Assigned': task,
        'Shift Availability': shift,
        'Performance Rating': performance_rating,
        'Certifications': certification
    }

# Generate 1000 worker entries
worker_data = [generate_worker_data(i) for i in range(1, 1001)]

# Create the DataFrame
worker_df = pd.DataFrame(worker_data)

# Save the dataset to a CSV file
worker_df.to_csv('worker_allocation_dataset_1000.csv', index=False)

# Display the path to the saved file
'worker_allocation_dataset_1000.csv'
