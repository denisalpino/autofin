from numpy import sin, cos, pi
from numpy.typing import NDArray
from pandas import DataFrame


def encode_cyclic(
    values: NDArray,
    col_name: str,
    max_val: int | NDArray
) -> DataFrame:
    """
    Encode cyclic features using sin and cos transformations.

    Parameters
    ---
    values : NDArray
        Array of values to encode
    col_name : str
        Base name for the output columns
    max_val : int | NDArray
        Maximum value for the cyclic feature (used for normalization)

    Returns
    ---
    DataFrame : DataFrame with sin and cos encoded features
    """
    encoded_features = DataFrame()
    encoded_features[f"{col_name}_sin"] = sin(2 * pi * values / max_val)
    encoded_features[f"{col_name}_cos"] = cos(2 * pi * values / max_val)
    return encoded_features
