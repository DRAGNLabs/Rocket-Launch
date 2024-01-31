# Rocket-Launch

Rocket-Launch is a generalized version of the Rocket framework. Rocket-Launch can use any HuggingFace model, is capable of using any HuggingFace dataset, and utilizes PyTorch Lightning to easily enable distributed training on a number of configurations. Rocket-Launch is designed to be a flexible research framework with the ability to:

- Finetune on any dataset.
- Train from scratch on any dataset.
- Enable users to modify low-level model code and architecture
- Scale up to large models with distributed training.

Rocket-Launch primarily uses HuggingFace and PyTorch Lightning to achieve these abilities. The user is encouraged to understand these tools. In short:

- HuggingFace easily provides a wide range of models and datasets to use.
- PyTorch Lightning enables high-performance distributed training, as well as great flexibility in training code setup for a variety of needs.

This repository assumes you are running the code on a Slurm-enabled supercomputing infrastructure, but this is not necessary.

## Project Structure

This repository consists of:

- **configs**: the configuration folder holding all configs for use in training, data preparation, and evaluation.
- **dataset**: the dataset folder should store all raw and tokenized data, as well as tokenizers.
- **data_setup**: contains scripts for downloading data, most notably from the HuggingFace Hub
- **runs**: contains all results from training and evaluation jobs.
- **slurm**: slurm scripts for various tasks.
- **tokenizer**: various scripts pertaining to tokenization, as well as the core tokenizer class in [tokenizer.py](./tokenizer/tokenizer.py).
- **utils**: various utils.
- **dataset.py**: containing PyTorch Lightning DataModule class and DataSet class. These classes should be modified for specific use cases.
- **generation.py**: script for generating from trained model.
- **inference.py**: script for running inference data on given metrics or benchmarks.
- **llama.py**: core LightningModule class for Llama.
- **model.py**: model code for Llama.
- **tokenize_data.py**: tokenizes data found in corresponding path in given config.
- **train.py**: training script.

## Workflow

A general workflow for this repository involves:

- Downloading a dataset to a data directory.
- Training a tokenizer on the data.
- Tokenizing the data with this tokenizer, and saving to the data directory.
- Training a model on the tokenized data.
- Running inference and/or generation with the trained model.

## Setting up Rocket Llama

### Environment

Create a Mamba environment with python=3.9, preferably named ```rocket```:
```mamba create -n rocket python=3.9```

If it is named differently, the environment activation commands in the Slurm scripts must be changed.

Run ```pip install -r requirements.txt```.

### Setting up a Config

Configuration YAML (YAML Ain't Markup Language) files are used to define all paths, settings, and hyperparameters for training tokenizers, tokenizing data, training models, and running inference on models. In the config folder, you can create a new config by copying default_config.yaml, preferebly into the [user_configs](./configs/user_configs/) folder. Fill out the class parameters accordingly.

- Any paths relating to the dataset or checkpoints should be in a directory with plenty of storage
- It's recommended to use absolute paths in the config.
- This repository is setup to work flexibly with any desired directory structure.
- This repository is setup to work flexibly with any dataset source. If retrieving datasets from the HuggingFace Hub, define the parameters to match.
- You may define paths for either one single dataset path, or seperate paths for train/test/eval dataset paths, depending on the form of the data.

### Setting up Slurm scripts

With the exception of downloading data, all steps in the pipeline are designed to be run through Slurm processes. The [slurm](./slurm/) folder contains default Slurm scripts for many steps in the pipeline. It is recommended to copy all necessary Slurm scripts into the [user_slurm](./slurm/user_slurm/) folder. Before running any Slurm script, edit the configuration to work for your usage. Ensure you are activating the right Mamba environment in the script, and that the correct config path is given.

### Getting Data

#### HuggingFace

To download datasets from the HuggingFace Hub, run [hf_data_setup.py](./hf_data_setup.py), passing in the path to your desired config file as a parameter. This will save the HF dataset as a parquet file in the given data folder.

#### OpenOrca

[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) is a logic and reasoning dataset. To obtain the OpenOrca data, run [orca_data_setup.py](./orca_data_setup.py). This will download two parquet files into your dataset folder. The script will then consolidate both parquet files into a single parquet file(for training), as well as a csv file(for training the tokenizer).

### Preparing Tokenizer

This repository is designed to work with either HuggingFace tokenizers or SentencePiece tokenizers.

#### Retrieiving or Training HuggingFace tokenizers

#### Training SentencePiece Tokenizer from scratch

A SentencePiece tokenizer can be trained by running [train_tokenizer.sh](./slurm/train_tokenizer.sh). This script is simply a wrapper for the SentencePiece python module; it seems easier than building and installing SentencePiece from source. Pass in all arguments in quotations, ex:

```python3 train_tokenizer.py "--input=../dataset/raw/openorca_combined.csv --input_format=text --input_sentence_size=1000000 --train_extremely_large_corpus=true --model_prefix=tokenizer --vocab_size=32000 --shuffle_input_sentence=true --pad_id=3""```

You can adjust the vocabularly size with `--vocab_size`.

You will want to verify that the [Tokenizer](./tokenizer/tokenizer.py) class is using ```.pad_id()``` as opposed to a custom pad string, i.e. "['<pad>']".

Then, submit the job:
```sbatch train_tokenizer.sh```

You can find further information on training arguments in the SentencePiece documentation: 
- [SentencePiece Repository](https://github.com/google/sentencepiece)
- [Training options](https://github.com/google/sentencepiece/blob/master/doc/options.md)

### Tokenizing data

To tokenize data, you will first want to create a new tokenization script. [wikitext_tokenization.py](./tokenizer/wikitext_tokenization.py) and [orca_tokenization.py](./tokenizer/orca_tokenization.py) provide good examples. How you tokenize is highly dependent on the data and it's structure.

Once you have the desired tokenization script, edit the import statement in [tokenizer_data.py](./tokenize_data.py) so that the correct tokenization script is being used.

Ensure that the correct tokenizer you wish to use is specified in the config file. Navigate to [tokenize_data.sh](./slurm/tokenize_data.sh) and verify that your desired config file is being passed as a parameter in the script. 

Then, submit the job:
```sbatch tokenize_data.sh```

[tokenize_data.py](./tokenize_data.py) will tokenize the given data files as defined in the config yaml file, according to the tokenizer path given. This script expects raw data to be in parquet file format by default, but this could be changed.

## Training

Before training, it may be desirable change the dataset processing in [dataset.py](./dataset.py). By default, the dataset class is padding each sequence in the batch. The best processing method is highly dependent on the data.

The [train.py](./train.py) takes as an argument a path to a config yaml file. There is a slurm script, [run_train.sh](./slurm/run_train.sh) that calls this script. Edit the slurm script to use your config file, and training will begin when ran.

## Inference