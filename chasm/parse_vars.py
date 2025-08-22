import numpy as np
import pandas as pd
import subprocess
import os

# Function to parse variables from the text file
def parse_variables(file_path):
    variables = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.strip() == '' or line.startswith('#'):
                continue
            # Split the line by '<-' to get variable name and value
            name, value = line.split('<-')
            # Remove leading and trailing whitespace from name and value
            name = name.strip()
            value = value.strip()
            # Convert value to appropriate data type
            try:
                value = int(value)  # Try converting to integer
            except ValueError:
                try:
                    value = float(value)  # Try converting to float
                except ValueError:
                    pass  # If conversion fails, keep it as string
            # Store variable in dictionary
            variables[name] = value
    return variables