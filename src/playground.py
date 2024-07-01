import subprocess
import time
import random
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Function to run a command in the console
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

# Function to format datetime for git
def format_git_date(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S')

client = openai.OpenAI()

def simulate_change(file_path=None):
    prompt = "You are a professional software developer. Generate Python code for a script that generates some fake data using the faker library for the purpose of economic analysis on antitrust litigation. Then usin statsmodels analyzes the fake data. In total uses libraries like pandas, numpy, faker, statsmodels and/or matplotlib for visualizations. The analysis is to be used for antitrust litigation. Be creative and Include only the python code in your response"

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
    generated_code = response.choices[0].message.content
    with open(file_path, 'a') as f:
        f.write(f"# Change made on {datetime.now()}\n")
        f.write(generated_code + "\n")
    return generated_code

# Function to generate a commit message using OpenAI API
def generate_commit_message(code):
    prompt = "You are a professional software developer. Generate a commit message for a git commit that includes new code. The code added was the following: "\
         + code
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
    commit_message = response.choices[0].message.content
    return commit_message

# Start date
current_date = datetime(2021, 12, 4)

# End date
end_date = datetime(2024, 3, 21)

while current_date < end_date:
    # Random hour of the day
    random_hour = random.randint(0, 23)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    
    # Set the time part to the random values
    current_date = current_date.replace(hour=random_hour, minute=random_minute, second=random_second)
    
    # Simulate a change in the file to ensure there is something to commit
    generated_code = simulate_change('src/antitrust.py')
    generated_code2 = simulate_change('src/empirical.py')
    
    # Generate commit message
    commit_message = generate_commit_message(generated_code)
    commit_command = f'GIT_AUTHOR_DATE="{format_git_date(current_date)}" GIT_COMMITTER_DATE="{format_git_date(current_date)}" git commit -m "{commit_message}"'
    print(format_git_date(current_date))
    # Run git commands
    run_command('git add -A')  # Stage all changes, including new files
    run_command(commit_command)
    
    # Check if there are any changes to stash
    status_result = subprocess.run('git status --porcelain', shell=True, capture_output=True, text=True)
    if status_result.stdout:
        run_command('git stash')
        run_command('git pull --rebase')
        run_command('git stash pop')
    else:
        run_command('git pull --rebase')
    
    run_command('git push')
    
    # Wait 1 minute
    #time.sleep(1)
    
    # Increment current_date by a random float number between 1 hour and 60 hours
    increment_hours = random.uniform(1, 200)
    current_date += timedelta(hours=increment_hours)
