from xgboost import XGBRegressor
import pandas as pd


class XGBoostModel:
    """
    Encapsulates the training logic with or without tuning using an XGBRegressor.
    """

    def __init__(self):
        self.model = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
        )

    def get_model_object(self):
        """
        Returns the model object.
        """
        return self.model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        hyperparameters_tuning: bool = False,
    ):
        if not hyperparameters_tuning:
            self.model.fit(X, y)
        else:
            raise NotImplementedError(
                "Hyperparameters tuning is not implemented for XGBoostModel"
            )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)
