from typing import Any, Dict

import numpy as np
import pandas as pd


class dataCleaner:
    """A tool to clean datasets for better training of models"""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Parameters
        ------------
        train: pd.DataFrame
            DataFrame that will be used to train the model

        test: pd.DataFrame
            DataFrame that will be used to evaluate the trained model
        """

        self.train: pd.DataFrame = train.copy()
        self.test: pd.DataFrame = test.copy()
        self.variables = train.columns.tolist()

    def show_duplicate_observations(
        self,
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """Returns a dict of duplicate rows between train and test"""

        train_duplicates = self.train.duplicated()
        test_duplicates = self.test.duplicated()

        return {
            "train": self.train[train_duplicates].copy(),
            "test": self.test[test_duplicates.copy()],
        }

    def remove_duplicate_observations(self) -> None:
        """Deletes duplicate rows"""

        self.train.drop_duplicates(inplace=True)
        self.test.drop_duplicates(inplace=True)

    def show_missing_values(self) -> Dict[str, int]:
        """Returns a dict containing the number of missing values in columns"""

        train_missing_cells = self.train.isna().sum()
        test_missing_cells = self.test.isna().sum()

        return {"train": train_missing_cells, "test": test_missing_cells}

    def fix_missing_values(
        self, feature: str = "", strategy: str = "Delete", n_neighbours: int = 3
    ) -> None:
        """
        Fixes the empty values in the datasets by the strategy of your choosing:
            KNN: Uses K Nearest Neigbours method to fill the empty cells
            Mean: Uses the mean value of the column to fill the empty cells
            Mode: Uses the most frequent element in the column to fill the empty cells
            Median: Uses the median value of the column to fill the empty cells
            Delete: Deletes the rows with empty values. Default

        --------------

        feature: string
            Column name to be fixed
        strategy: string
            Strategy of fixing
        n_neighbours: integer
            Number of closest neighbours to check in KNN strategy
        """

        train_notna_idx = None
        test_na_idx = None
        train_na_idx = None

        if strategy != "KNN":
            if not feature:
                raise ValueError("Your strategy requires a column name to be provided")

            train_notna_idx = self.train.loc[:, feature].notna().values
            test_na_idx = self.train.loc[:, feature].isna().values
            train_na_idx = self.train.loc[:, feature].isna().values

        if strategy == "Delete":
            self.train = self.train.drop(columns=feature)
            self.test = self.test.drop(columns=feature)

        elif strategy == "Mean":
            mean = self.train.loc[train_notna_idx, feature].mean()
            print("Mean: ", mean)
            self.train.loc[train_na_idx, feature] = mean
            self.test.loc[test_na_idx, feature] = mean

        elif strategy == "Mode":
            mode = self.train.loc[train_notna_idx, feature].mode()[0]
            print("Mode: ", mode)
            self.train.loc[train_na_idx, feature] = mode
            self.test.loc[test_na_idx, feature] = mode

        elif strategy == "Median":
            median = self.train.loc[train_notna_idx, feature].median()
            print("Mode: ", median)
            self.train.loc[train_na_idx, feature] = median
            self.test.loc[test_na_idx, feature] = median

        else:
            raise ValueError("Are you sure you chose the strategy correctly")

    def outlier_detection(
        self,
        feature: str = "",
        strategy: str = "inter_quartile_range",
        n_estimators: int = 50,
        contamination: float | str = "auto",
        max_samples: float | str = "auto",
    ) -> Dict[str, Any]:
        """
        Finds the outliers

        Args:
            feature (str, optional): column name. Defaults to "".
            strategy (str, optional): strategy for outlier_detection. Defaults to "inter_quartile_range".
            n_estimators (int, optional): Defaults to 50.
            contamination (float | int, optional): Defaults to 0.1.
        """

        train_outliers = None
        test_outliers = None

        if strategy == "inter_quartile_range":
            Q1 = self.train.loc[:, feature].quantile(0.25)
            Q3 = self.train.loc[:, feature].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            print("Upper Limit: ", upper_limit, "\nLower Limit: ", lower_limit)

            train_out = np.array(
                (self.train[feature] < lower_limit)
                & (self.train[feature] > upper_limit)
            )

            test_out = np.array(
                (self.test[feature] < lower_limit) & (self.test[feature] > upper_limit)
            )

            train_outliers = self.train.loc[train_out]
            test_outliers = self.test.loc[test_out]

        else:
            raise ValueError("Wrong method choice")

        return {"train": train_outliers, "test": test_outliers}
