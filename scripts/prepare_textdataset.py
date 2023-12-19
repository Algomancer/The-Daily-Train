# prepare_textdataset.py
"""
This script is a generalized module for preparing text datasets by tokenizing the text and saving it to a binary file.
It is designed to work with any text dataset available in the Hugging Face datasets library.
"""

import os
import sys
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Ensure the daily_train package is available
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from daily_train import Tokenizer

def print_dataset_structure(dataset_name: str, split: str = 'train', entries: int = 5):
    """
    Prints the structure of the specified dataset by showing the first few entries.

    Args:
        dataset_name (str): The name of the dataset to print.
        split (str): The dataset split to print.
        entries (int): The number of entries to print from the dataset.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for i, example in enumerate(dataset):
        if i >= entries:
            break
        print(f"Entry {i + 1}:")
        for key, value in example.items():
            print(f"  {key}: {value}")

def process(example: Dict[str, Any], text_key: str, tokenizer: Tokenizer) -> Dict[str, Any]:
    """
    Tokenizes the text of an example and appends the end-of-sequence token.

    Args:
        example (Dict[str, Any]): A dictionary containing the fields of a dataset example.
        text_key (str): The key corresponding to the text field in the example.
        tokenizer (Tokenizer): An instance of the Tokenizer class.

    Returns:
        Dict[str, Any]: A dictionary with the tokenized 'ids' and their 'len'.
    """
    ids = tokenizer.encode(example[text_key]).tolist()
    ids.append(tokenizer.eos_id)
    return {"ids": ids, "len": len(ids)}

def prepare(
    dataset_name: str,
    text_key: str,
    destination_path: Path,
    checkpoint_dir: Path,
    seed: int = 42,
    test_size: Union[float, int, None] = 0.0005,
) -> None:
    """
    Prepares a text dataset for training by tokenizing and saving it to a binary file.

    Args:
        dataset_name (str): The name of the dataset to prepare.
        text_key (str): The key of the text field in the dataset.
        destination_path (Path): The directory where the tokenized dataset will be saved.
        checkpoint_dir (Path): The directory containing the tokenizer checkpoint.
        seed (int): The seed for random operations to ensure reproducibility.
        test_size (Union[float, int, None]): The proportion or absolute number of dataset samples to include in the test split.
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(checkpoint_dir)
    num_proc = os.cpu_count() // 2

    # Load the dataset
    dataset = load_dataset(dataset_name, num_proc=num_proc)

    # Check if the dataset has a 'train' split
    if "train" not in dataset.keys():
        print(f"Warning: The dataset '{dataset_name}' does not have a 'train' split.")
        return

    # Check if the text_key exists in the dataset
    if text_key not in dataset["train"].column_names:
        print(f"Warning: The key '{text_key}' does not exist in the dataset '{dataset_name}'. Please check the dataset structure.")
        return

    # Print metadata about the dataset
    print(f"Dataset '{dataset_name}' metadata:")
    print(dataset["train"].info.description)
    print(f"Number of documents: {dataset['train'].num_rows}")
    print(f"Size: {dataset['train'].info.size_in_bytes / 1024**2:.2f} MB")

    # Create a test split from the 'train' split
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # Tokenize the dataset
    tokenized = split_dataset.map(lambda example: process(example, text_key, tokenizer), remove_columns=[text_key], desc="tokenizing the splits", num_proc=num_proc)
    
    # Save the tokenized data to a binary file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = destination_path / f"{split}.bin"
        dtype = np.uint16  # Can use since tokenizer's max_token_value is < 2**16
        arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == "__main__":
    from jsonargparse import CLI, ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to load")
    parser.add_argument("text_key", type=str, help="Key of the text field in the dataset")

    CLI(prepare)