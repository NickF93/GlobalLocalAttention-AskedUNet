import json
from typing import Dict, Any

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the configuration.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from file: {file_path}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e
