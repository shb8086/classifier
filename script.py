import subprocess
import os
import sys
import joblib
import pandas as pd

# Install the required packages using pip if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL)
print("Packages installed successfully!")

# Loaded data preprocessing
def preprocess_data(df):
  try:
    bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    df["age"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    bins=[0, 10, 20, 30, 100]
    labels=[1, 2, 3, 4]
    df["Na_to_K_rate"] = pd.cut(df["Na_to_K_rate"], bins=bins, labels=labels, right=False)

    df = pd.get_dummies(df, drop_first=True)

  except Exception as e:
    print(f"Error: {e}")
  
  return df

# Execute the Python script and capture its output
def execute_python_script(script_file_path, output_file_path):
  try:
    # Execute the Python script and capture its output
    output = subprocess.check_output(['python', script_file_path], stderr=subprocess.STDOUT, text=True)
    # Write the output to a text file
    with open(output_file_path, 'w') as f:
      f.write(output)
    print(f"Output written to {output_file_path}")

  except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

if __name__ == "__main__":

  # create folders if they don't exist
  if not os.path.exists('output/plots'):
    os.makedirs('output/plots')
  if not os.path.exists('dataset/preprocessed'):
    os.makedirs('dataset/preprocessed')
  if not os.path.exists('models'):
    os.makedirs('models')

  # Check if the preprocessed dataset exists
  if not os.path.exists('dataset/preprocessed/X_train.npy'):
    python_script_path = 'src/preprocessing.py'
    output_file_path = 'output/preprocessing.out'
    execute_python_script(python_script_path, output_file_path)
  else:
    print("The preprocessed dataset already exists. Skipping preprocessing.")

  # Check if the classifier model exists
  if not os.path.exists('models/classifier_model.pkl'):
    python_script_path = 'src/classifier.py'
    output_file_path = 'output/classifier.out'
    execute_python_script(python_script_path, output_file_path)
  else:
    print("The classifier model already exists. Skipping model training.")