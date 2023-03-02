
class ProcessConfig:
    """Specify the parameters of the `process` flow"""

    bankruptcy_label: str = "Bankrupt?"
    
    creditcard_label: str = "Class"
    
    diabetes_label: str = "Outcome"
    
    churn_label: str = "Churn"
    
    test_size: float = 0.2