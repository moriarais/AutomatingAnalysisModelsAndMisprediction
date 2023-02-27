
# data processing
import os
import sys

"""Specify the locations of inputs and outputs"""

# Get the path of the directory containing the script file
script_dir = os.path.dirname(os.path.abspath(sys.path[0]))

# Navigate up to the top-level directory
src_level_dir = os.path.dirname(script_dir)

top_level_dir = os.path.dirname(src_level_dir)

# Processing configuration
class Location():
    # Define the relative path to the data directory
    data_dir = os.path.join(top_level_dir, "AutomatingAnalysisModelsAndMisprediction\\data")
    
    company_bankruptcy_data_raw: str = f"{data_dir}\\raw\\company_bankruptcy.csv"
    company_bankruptcy_data_process: str = f"{data_dir}\\processed\\company_bankruptcy.pkl"
    
    creditcard_data_raw: str = f"{data_dir}\\raw\\creditcard.csv"
    creditcard_data_process: str = f"{data_dir}\\processed\\creditcard.pkl"
    
    health_data_raw: str = f"{data_dir}\\raw\\health_diabetes.csv"
    health_data_process: str = f"{data_dir}\\processed\\health_diabetes.pkl"
    
    customer_churn_data_csv_process: str = f"{data_dir}\\processed\\customer_churn.csv"
    customer_churn_data_process: str = f"{data_dir}\\processed\\customer_churn.pkl"
    
    
    
class ProcessConfig:
    """Specify the parameters of the `process` flow"""

    company_bankruptcy_label: str = "Bankrupt?"
    
    creditcard_label: str = "Class"
    
    health_label: str = "Outcome"
    
    churn_label: str = "Churn"
    
    test_size: float = 0.2