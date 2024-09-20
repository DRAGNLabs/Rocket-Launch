import argparse
import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from pathlib import Path
import pyarrow as pa
import yaml

from transformers import PreTrainedTokenizerFast as HFTokenizer

from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from utils.data_utils import Struct

ProgressBar().register()

def generate_tokenized_file(raw_data_path: Path, tokenizer_path, tokenizer_type, dataset_feature):
    """
    Tokenizes a dataset, returning a DataFrame with the tokenized data.

    This function is highly dependent on the structure of the input data, and the tokenizer being used.
    """
    # Load Dataset into pd.DataFrame
    dataset = dd.read_parquet(raw_data_path, dtype=str, na_filter=False)
    
    # Load tokenizer
    if tokenizer_type == 'hf':
        tokenizer = HFTokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_type == 'sp':
        tokenizer = SPTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Tokenizer type '{tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")

    def tokenize_partition(partition):
        tokenization_dataframe = lambda series: \
            tokenizer.encode(
                series,
                bos=True, 
                eos=True)

        tokenized_data = partition[dataset_feature] \
            .map(tokenization_dataframe, na_action='ignore').to_frame()

        return tokenized_data
    
    dataset = dataset.map_partitions(tokenize_partition)

    return dataset

def tokenize_data(config: Struct, split: str):
    print('\nStarting tokenization...\n')
    dataset_feature = config.dataset_feature if config.dataset_feature else 'data'

    data_path = Path(config.raw_dataset_path) / split / "*.parquet"

    # Generate tokenized file
    dataset = generate_tokenized_file(data_path, 
                                        tokenizer_path=config.tokenizer_path, 
                                        tokenizer_type=config.tokenizer_type,
                                        dataset_feature=dataset_feature)

    # Save train, validation, and test to pickle files
    tokenized_data_dir = Path(config.tokenized_dataset_path)
    tokenized_data_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenized data to {config.tokenized_dataset_path}") 
    
    dataset.to_parquet(tokenized_data_dir / split, 
                             schema={dataset_feature: pa.list_(pa.int64())})

    print('Done!')

def main():
    parser = argparse.ArgumentParser(description='Tokenize data')
    parser.add_argument('config_path', 
                        type=str, 
                        help='Path to the config file')
    parser.add_argument('split', 
                        type=str, 
                        choices=['train', 'test', 'validation'], 
                        help='Dataset split to use')
    args = parser.parse_args()

    config_path = args.config_path
    split = args.split

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    tokenize_data(config, split)

if __name__== "__main__":
    main()
