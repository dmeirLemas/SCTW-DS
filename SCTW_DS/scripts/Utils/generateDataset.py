from typing import Callable, List, Tuple

import numpy as np
import pandas as pd


def balance_classes(df, y_col):
    # Group the data by the class column
    grouped = df.groupby(y_col)

    # Find the size of the smallest class
    min_size = grouped.size().min()

    # Resample all classes to match the size of the smallest class
    balanced_groups = [group.sample(min_size, random_state=42) for _, group in grouped]

    # Concatenate the resampled classes back together
    balanced_data = pd.concat(balanced_groups)

    return balanced_data


def generateDataset(
    name: str,
    num_input_variables: int,
    x_funcs: List[Callable[..., np.ndarray]],
    x_func_params: List[Tuple],
    y_func: Callable[[List[np.ndarray]], np.ndarray],
    balance_dataset: bool = True,
) -> None:
    def generate_data():
        input_variables: List[np.ndarray] = []
        for i in range(num_input_variables):
            input_col = x_funcs[i](*x_func_params[i])
            input_variables.append(input_col)

        y_col = y_func(input_variables)

        temp = {f"input_{i}": input_variables[i] for i in range(len(input_variables))}
        temp["y_col"] = y_col

        return pd.DataFrame.from_dict(temp)

    df = generate_data()

    # Determine the required size
    required_size = x_func_params[0][-1]

    if balance_dataset:
        balanced_df = balance_classes(df, "y_col")

        # Keep generating data and balancing until we have enough samples
        while len(balanced_df) < required_size:
            new_df = generate_data()
            balanced_new_df = balance_classes(new_df, "y_col")
            balanced_df = pd.concat([balanced_df, balanced_new_df])

        # Shuffle and sample to get the exact required size
        balanced_df = balanced_df.sample(n=required_size, random_state=42).reset_index(
            drop=True
        )
        balanced_df.to_csv(name, index=False)
    else:
        df = df.sample(n=required_size, random_state=42).reset_index(drop=True)
        df.to_csv(name, index=False)


# Example usage (assuming x_funcs, x_func_params, and y_func are defined)
# generateDataset('output.csv', num_input_variables, x_funcs, x_func_params, y_func)
