from typing import Optional
import pandas as pd


class DummyModel:
    """
    A dummy model that predicts the last known close price as the next close price
    N minutesto in the future.
    """

    def __init__(self, from_feature: Optional[str] = "close"):
        """
        Initilaize the dummy model. We store the name of the feature we will use as the prediction.

        Args:
            from_features: The name of the feature we will use as the prediction.
        """
        self.from_feature = from_feature

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the next close price N minutes in the future.
        """

        try:
            return data[self.from_feature]
        except KeyError:
            raise ValueError(f"Feature {self.from_feature} not found in data")

        return data
