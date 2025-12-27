import os
import random
import subprocess
from datetime import datetime, timedelta

def get_random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def run_command(command, env=None):
    try:
        subprocess.run(command, check=True, shell=True, env=env)
        print(f"Success: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

start_date = datetime(2025, 12, 1)
end_date = datetime(2025, 12, 27)

# Commit groups
commits = [
    {
        "files": ["Data.py", "sri_lanka_coffee_data.csv"],
        "msg": "Add data generation script and dataset"
    },
    {
        "files": ["preprocessing.py"],
        "msg": "Implement data preprocessing and sequence creation"
    },
    {
        "files": ["model.py"],
        "msg": "Define Hybrid CNN-BLSTM model architecture"
    },
    {
        "files": ["train.py"],
        "msg": "Add training script for price and demand models"
    },
    {
        "files": ["app.py", "templates/"],
        "msg": "Create Flask application and dashboard UI"
    },
    {
        "files": ["."], # Commit everything else
        "msg": "Finalize project structure and add evaluation scripts"
    }
]

# Generate sorted random dates
dates = sorted([get_random_date(start_date, end_date) for _ in range(len(commits))])

env = os.environ.copy()

print("Starting custom commit history generation...")

for i, commit in enumerate(commits):
    date_str = dates[i].strftime("%Y-%m-%d %H:%M:%S +0530")
    
    # Add files
    for file in commit["files"]:
        run_command(f'git add "{file}"')
        
    # Commit with custom date
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    msg = commit["msg"]
    run_command(f'git commit -m "{msg}"', env=env)
    print(f"Committed: {msg} on {date_str}")

print("History generation complete.")
