import sys
import yaml

from pathlib import Path
from utils.data_utils import Struct

import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
import dask_ml

def split_data(config):
    """ 
    Filter and split the dataset into training, validation, and testing datasets.
    """
    # Create folder to save this dataset's files in
    dataset_dir = Path(config.raw_dataset_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create folder to save the split datasets in
    train_path = dataset_dir / "train"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path = dataset_dir / "validation"
    val_path.mkdir(parents=True, exist_ok=True)
    test_path = dataset_dir / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Read the dataset from disk
    dataset = dd.read_csv(dataset_dir / "*.csv", header=None, usecols=[0], names=['data'])

    # Split into training, validation, and testing datasets
    train, test_valid = dask_ml.model_selection.train_test_split(
        dataset,
        shuffle=True, # Very expensive for large datasets
        train_size=config.splits[0],
        random_state=config.seed)
    test, validation = dask_ml.model_selection.train_test_split(
        test_valid,
        shuffle=False,
        train_size=config.splits[1] / (config.splits[1] + config.splits[2]),
        random_state=config.seed)

    train.to_parquet(train_path)
    validation.to_parquet(val_path)
    test.to_parquet(test_path)

    print("Finished")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    split_data(config)