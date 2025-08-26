import numpy as np
import pandas as pd
import subprocess
import os
from typing import Dict, Tuple

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

def load_config(path_vars) -> Dict[str, int]:
    """Load scalar variables (G, L, c, k, M, etc.) from a text file parsed
    by your existing `parse_variables`. Falls back to environment defaults.
    """
    if parse_variables is None:
        raise RuntimeError(
            "`parse_variables` not available. Ensure scripts/parse_vars.py is importable."
        )
    variables = parse_variables(str(path_vars))
    # Cast all numeric-looking entries to int when possible
    cfg: Dict[str, int] = {}
    for k, v in variables.items():
        try:
            cfg[k] = float(v)
        except (TypeError, ValueError):
            # keep as-is if not int-like
            cfg[k] = v
    required = ["G", "L", "c", "k", "M", "F","very_rare_threshold_L","very_rare_threshold_H"]
    missing = [x for x in required if x not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg