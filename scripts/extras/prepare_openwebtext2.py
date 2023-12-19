# prepare_openwebtext2.py
"""
This script prepares the OpenWebText2 dataset for training by tokenizing the text and saving it to a binary file.
OpenWebText2 is an open-source replication of the WebText dataset used by OpenAI for training language models.
It contains a large number of web documents that are useful for unsupervised learning of language models.

Key features of the OpenWebText2 dataset:
- It is larger and more diverse than the original OpenWebText.
- This script uses the Plug and Play version of the dataset, which has been deduped by URL and MinHashLSH.
- Contains millions of documents sourced from URLs shared on Reddit with at least 3 upvotes.
- The dataset is cleaned and deduplicated.
- Size: 65.86 GBApproximately 63GB of text data.
- Number of documents: Over 20 million.

The script uses the Hugging Face `datasets` library to load the dataset and the `Tokenizer` class from the
`daily_train` package to tokenize the text. The tokenized data is then saved in a memory-mapped binary file
for efficient reading during training.
"""

import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from daily_train import Tokenizer


def prepare(
    destination_path: Path = Path("data/openwebtext2"),
    checkpoint_dir: Path = Path("checkpoints/codellama/CodeLlama-7b-Python-hf"),
    seed: int = 42,
    test_size: Union[float, int, None] = 0.0005,
) -> None:
    """
    Prepares the OpenWebText2 dataset for training by tokenizing and saving it to a binary file.

    Args:
        destination_path (Path): The directory where the tokenized dataset will be saved.
        checkpoint_dir (Path): The directory containing the tokenizer checkpoint.
        seed (int): The seed for random operations to ensure reproducibility.
        test_size (Union[float, int, None]): The proportion or absolute number of dataset samples to include in the test split.
    """
    from datasets import load_dataset  # huggingface datasets

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    num_proc = os.cpu_count() // 2
    num_proc_load_dataset = num_proc

    # Load the OpenWebText2 dataset
    dataset = load_dataset("openwebtext2", num_proc=num_proc_load_dataset)

    # Create a test split from the 'train' split
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        """
        Tokenizes the text of an example and appends the end-of-sequence token.

        Args:
            example (dict): A dictionary containing the 'text' field of a dataset example.

        Returns:
            dict: A dictionary with the tokenized 'ids' and their 'len'.
        """
        ids = tokenizer.encode(example["text"]).tolist()
        ids.append(tokenizer.eos_id)
        return {"ids": ids, "len": len(ids)}

    # Tokenize the dataset
    tokenized = split_dataset.map(process, remove_columns=["text"], desc="tokenizing the splits", num_proc=num_proc)

    # Save the tokenized data to a binary file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = destination_path / f"{split}.bin"
        dtype = np.uint16
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
    from jsonargparse import CLI

    CLI(prepare)