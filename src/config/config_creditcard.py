"""
create Pydantic models
"""
from typing import List
class Location():
    """Specify the locations of inputs and outputs"""

    data_raw: str = f"data/raw/creditcard.csv"
    data_process: str = "data/processed/creditcard.pkl"
    data_final: str = "data/final/predictions-creditcard.pkl"
    model: str = "models/svc-creditcard.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"

class ProcessConfig():
    """Specify the parameters of the `process` flow"""

    label: str = "Class"
    test_size: float = 0.2
