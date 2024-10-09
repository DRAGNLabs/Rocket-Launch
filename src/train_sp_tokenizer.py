import sentencepiece as spm
import shutil
import sys
from pathlib import Path
import yaml

import dask
dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
from utils.data_utils import Struct

def train_tokenizer(config):

    dataset = dd.read_parquet(Path(config.raw_dataset_path) / "train" / "*.parquet")
    samples = dataset.sample(frac=0.2).compute()["data"]

    # Output to csv to be used as training data from SP
    samples.to_csv("samples.csv", index=False, header=False)

    # TODO: put this and other args in config
    spm.SentencePieceTrainer.train(input="samples.csv",
                                input_format="text",
                                model_prefix="sp",
                                pad_id=config.pad_id,
                                bos_id=2,
                                eos_id=3,
                                vocab_size=config.vocab_size,
                                character_coverage=1.0,
                                model_type="unigram",
                                # user_defined_symbols=["+", "=", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
                                )

    # Move .model and .vocab to Tokenizers folder
    for file in Path().glob("*.model"):
        dest_path = Path(config.tokenizer_path)
        shutil.move(str(file), dest_path)

    for file in Path().glob("*.vocab"):
        dest_path = Path(config.vocab_path)
        shutil.move(str(file), dest_path)

    # Clean up
    Path("samples.csv").unlink()

    print("Finished")


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    train_tokenizer(config)
