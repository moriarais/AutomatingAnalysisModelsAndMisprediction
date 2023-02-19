"""
create Pydantic models
"""
from typing import List
from wsgiref.validate import validator



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

    drop_columns: List[str] = ["Id"]
    label: str = "Species"
    test_size: float = 0.3

    # _validated_test_size = validator("test_size", allow_reuse=True)(
    #     must_be_non_negative
    # )


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v



class ProcessConfig():
    """Specify the parameters of the `process` flow"""

    drop_columns: List[str] = ["Id"]
    label: str = "Species"
    test_size: float = 0.3

    # _validated_test_size = validator("test_size", allow_reuse=True)(
    #     must_be_non_negative
    # )


# class ModelParams():
#     """Specify the parameters of the `train` flow"""

#     C: List[float] = [0.1, 1, 10, 100, 1000]
#     gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]

#     _validated_fields = validator("*", allow_reuse=True, each_item=True)(
#         must_be_non_negative
#     )
