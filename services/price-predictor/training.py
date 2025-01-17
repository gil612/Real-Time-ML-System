from loguru import logger


def train(days_back: int):
    """
    Does the following:
    1. Reads feature dsata from the Feature Store
    2. Splits the data into training and testing sets
    3. Trains the model on the training set
    4. Evaluates the model on the testing set
    5. Saves the model to the model registry

    Everything is instrumented with CometML
    """
    logger.info("Hello from the ML model training job")

    # 1. Read feature data from the Feature Store
    # feature_reader = FeatureReader()
    # feature_reader.get_training_data(days_back=days_back)
    # 2. Split the data into training and testing sets

    # 3. Train the model on the training set

    # 4. Evaluate the model on the testing set

    # 5. Save the model to the model registry


if __name__ == "__main__":
    from config import training_config

    train(training_config.days_back)
