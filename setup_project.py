import os
import sys

# Define the directory structure as a dictionary
structure = {
    "data": ["raw", "processed", "external"],
    "models": [],
    "utils": [],
    "experiments": ["experiment_1", "experiment_2"],
    "config": [],
    "notebooks": [],
    "scripts": [],
    "tests": [],
}

# List of files to be created at the root level of the project
root_files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "main.py"
]

# List of files to be created inside each directory
dir_files = {
    "models": [
        "base_model.py",
        "dqn_model.py",
        "rnn_model.py",
        "any_task_learner.py",
        "__init__.py"
    ],
    "utils": [
        "data_processing.py",
        "model_utils.py",
        "metrics.py",
        "visualization.py"
    ],
    "config": [
        "config.yaml",
        "dqn_config.yaml",
        "rnn_config.yaml",
        "any_task_config.yaml"
    ],
    "notebooks": [
        "data_exploration.ipynb",
        "model_training.ipynb",
        "model_evaluation.ipynb"
    ],
    "scripts": [
        "train.py",
        "evaluate.py",
        "deploy.py"
    ],
    "tests": [
        "test_models.py",
        "test_utils.py",
        "test_data.py",
        "__init__.py"
    ]
}

# Function to create the directory structure
def create_project_structure(base_path, structure, root_files, dir_files):
    for subdir in structure:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

        for subsubdir in structure[subdir]:
            subdir_path = os.path.join(dir_path, subsubdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Created directory: {subdir_path}")

        if subdir in dir_files:
            for file_name in dir_files[subdir]:
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'w') as file:
                    file.write(f"# {file_name.split('.')[0]}")
                print(f"Created file: {file_path}")

    for file_name in root_files:
        file_path = os.path.join(base_path, file_name)
        with open(file_path, 'w') as file:
            file.write(f"# {file_name.split('.')[0]}")
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    # Allow the user to specify the project name as an argument
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "any-task-learner"

    # Create the project structure
    create_project_structure(project_name, structure, root_files, dir_files)
