import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import pandas as pd

from .progress_bar import ProgressBar


def process_chunk(
    df_chunk: pd.DataFrame,
    cols: pd.Index,
    p: ProgressBar,
    lock: threading.Lock,
    num_out_classes: int,
) -> np.ndarray:
    data_points = []
    for _, row in df_chunk.iterrows():
        inputs = [row[cols[j]] for j in range(len(cols) - 1)]
        output = row[cols[-1]]
        expected_outputs = [0.0] * num_out_classes
        expected_outputs[int(output)] = 1.0
        data_points.append(
            (
                np.array(inputs, dtype=np.float64),
                np.array(expected_outputs, dtype=np.float64),
            )
        )
        with lock:
            p.increment()

    # Convert list of tuples to a structured numpy array
    data_points = np.array(data_points, dtype=object)
    return data_points


def trainTestSplit(
    df: pd.DataFrame, num_out_classes: int, ratio: float = 0.8, num_threads: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
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

    return np.array(train_data), np.array(test_data)
