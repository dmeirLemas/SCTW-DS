import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd

from .progress_bar import ProgressBar


def process_chunk(
    df_chunk: pd.DataFrame,
    cols: pd.Index,
    p: ProgressBar,
    lock: threading.Lock,
    num_out_classes: int,
) -> List[Tuple[List[float], List[float]]]:
    data_points = []
    for _, row in df_chunk.iterrows():
        inputs = [float(row[cols[j]]) for j in range(len(cols) - 1)]
        output = row[cols[-1]]
        expected_outputs = [0.0] * num_out_classes
        expected_outputs[int(output)] = 1.0
        data_points.append(
            (
                inputs,
                expected_outputs,
            )
        )

        with lock:
            p.increment()

    return data_points


def trainTestSplit(
    df: pd.DataFrame, num_out_classes: int, ratio: float = 0.8, num_threads: int = 4
) -> Tuple[
    List[Tuple[List[float], List[float]]], List[Tuple[List[float], List[float]]]
]:
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    data_points = []
    cols = df.columns
    p = ProgressBar(total=len(df), program_name="PAIN")
    lock = threading.Lock()

    chunk_size = len(df) // num_threads
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk, cols, p, lock, num_out_classes): chunk
            for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            data_points.extend(future.result())

    train_data = data_points[: int(ratio * len(data_points))]
    test_data = data_points[int(ratio * len(data_points)) :]

    for i in range(len(test_data)):
        try:
            # Check the shape and content of expected_outputs
            if len(test_data[i][1]) == 0 or np.sum(test_data[i][1]) == 0:
                raise ValueError(
                    f"Expected output array has no 1.0 value: {test_data[i][1]}"
                )

            # Ensure the expected output remains in the one-hot encoded format
            test_data[i] = (
                test_data[i][0],
                test_data[i][1].index(1.0),  # Keep the one-hot encoded format
            )
        except IndexError as e:
            print(
                f"IndexError: {e} at index {i} with expected_outputs: {test_data[i][1]}"
            )
            raise
        except ValueError as e:
            print(f"ValueError: {e} at index {i}")
            raise

    return train_data, test_data
