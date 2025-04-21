import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from biotite.sequence.io.fasta import FastaFile

from augmented_potts.constants import AA_ALPHABET_GREMLIN


def format_a2m(msa_path: str) -> str:
    file_name = Path(msa_path).name
    directory = Path(msa_path).parent
    formatted_msa_path = f"{directory}/modified_{file_name}"
    with open(msa_path, "r") as rf:
        with open(formatted_msa_path, "w") as wf:
            for line in rf:
                if line.startswith(">"):
                    wf.write(line)
                else:
                    wf.write(line.upper().replace(".", "-"))
    return formatted_msa_path


def parse_mrf(mrf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(mrf_path, "r") as f:
        v_dict = {}
        w_dict = {}
        for line in f:
            split_line = line.split()
            mrf_id = split_line[0]
            if line.startswith("V"):
                v_dict[mrf_id] = list(map(float, split_line[1:]))
            elif line.startswith("W"):
                w_dict[mrf_id] = list(map(float, split_line[1:]))
    sequence_length = len(v_dict)
    v = np.zeros((sequence_length, 21))
    w = np.zeros((sequence_length, sequence_length, 21, 21))
    for key, value in v_dict.items():
        v_idx = int(*re.findall(r"\d+", key))
        v[v_idx] = value
    for key, value in w_dict.items():
        w_idx = list(map(lambda x: int(x), re.findall(r"\d+", key)))
        w[tuple(w_idx)] = np.reshape(value, (21, 21))
    return v, w


def tokenize_gremlin(letter: str) -> int:
    return AA_ALPHABET_GREMLIN.index(letter)


def encode_sequences_one_hot(sequences: Union[List[str], pd.Series]) -> np.ndarray:
    return np.array(
        [
            np.array(
                [
                    np.eye(len(AA_ALPHABET_GREMLIN))[AA_ALPHABET_GREMLIN.index(aa)]
                    for aa in sequence
                ]
            )
            for sequence in sequences
        ]
    )


def get_dummy_msa(sequence_length: int, n_hits: int = 100) -> FastaFile:
    if FastaFile is None:
        raise ImportError
    msa_fasta_file = FastaFile()
    msa_grid = np.random.choice(AA_ALPHABET_GREMLIN, (n_hits, sequence_length))
    msa_sequences = ["".join(x) for x in msa_grid]
    for i, sequence in enumerate(msa_sequences):
        msa_fasta_file[str(i)] = sequence
    return msa_fasta_file


def write_fasta(
    path: Union[str, Path], sequences: Union[str, FastaFile, Dict, List]
) -> None:
    file = FastaFile()
    path = str(path)
    if isinstance(sequences, str):
        sequences = read_fasta(sequences)
    if isinstance(sequences, list):
        for i, sequence in enumerate(sequences):
            file[str(i)] = sequence
    elif isinstance(sequences, dict):
        for key, sequence in sequences.items():
            file[key] = sequence
    elif isinstance(sequences, FastaFile):
        file = sequences
    else:
        raise TypeError("Sequences are not of type FastaFile, Dict or List")
    file.write(path)
