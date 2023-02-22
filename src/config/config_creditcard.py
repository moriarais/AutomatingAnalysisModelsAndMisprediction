"""
create Pydantic models
"""
import os

class Location:
    """Specify the locations of inputs and outputs"""

    # Get the path of the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the top-level directory
    src_level_dir = os.path.dirname(script_dir)

    top_level_dir = os.path.dirname(src_level_dir)

    # Define the relative path to the data directory
    data_dir = os.path.join(top_level_dir, "data")

    data_raw: str = f"{data_dir}\\raw\\creditcard.csv"
    data_process: str = f"{data_dir}\\processed\\creditcard.pkl"
    model: str = "models/svc-creditcard.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"

class ProcessConfig:
    """Specify the parameters of the `process` flow"""

    label: str = "Class"
    test_size: float = 0.2
