import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from augmented_potts.utils import (
    encode_sequences_one_hot,
    format_a2m,
    parse_mrf,
    tokenize_gremlin,
)


class PottsModel:
    """Direct coupling analysis model for generating evolutionary embeddings and
    calculating sequence energies"""

    def __init__(self, hi: np.ndarray = None, jij: np.ndarray = None):
        self.hi = hi
        self.jij = jij

    def run_gremlin(self, msa_path: str) -> None:
        if msa_path.endswith("a2m"):
            msa_path = format_a2m(msa_path)
        with TemporaryDirectory() as temp_dir:
            output_mrf_path = f"{temp_dir}/output.mrf"
            _ = subprocess.run(
                [
                    "gremlin_cpp",
                    "-i",
                    msa_path,
                    "-mrf_o",
                    output_mrf_path,
                    "-o",
                    f"{temp_dir}/ouputs",
                    "-gap_cutoff",
                    "1",
                    "-max_iter",
                    "100",
                ]
            )
            self.hi, self.jij = parse_mrf(output_mrf_path)

    def embed(self, sequences: List[str]) -> np.ndarray:
        tokenized_sequences = []
        for seq in sequences:
            seq = list(map(tokenize_gremlin, seq))
            tokenized_sequences.append(seq)
        seqs = np.array(tokenized_sequences)
        sequence_length = seqs.shape[1]
        embeddings = []
        for pos1 in range(sequence_length):
            hi_features = self.hi[pos1, seqs[:, pos1]].reshape(1, -1)
            jij_features = []
            for pos2 in range(sequence_length):
                jij_ = self.jij[pos1, pos2, seqs[:, pos1], seqs[:, pos2]]
                jij_features.append(jij_)
            jij_features = np.array(jij_features)
            emission_features = np.vstack((hi_features, jij_features))
            embeddings.append(emission_features)
        embeddings = np.transpose(np.array(embeddings), (2, 0, 1))
        return embeddings

    def predict(self, sequences: Union[pd.DataFrame, List[str]]) -> np.ndarray:
        if isinstance(sequences, pd.DataFrame):
            sequences = sequences["sequence"]
        embeddings = self.embed(sequences)
        predictions = self._calculate_sequence_energy(embeddings)
        return predictions

    @staticmethod
    def _calculate_sequence_energy(embeddings) -> np.ndarray:
        return np.sum(embeddings, axis=(1, 2))

    @classmethod
    def load(cls, model_directory: Union[Path, str]) -> "PottsModel":
        instance = cls(
            hi=np.load(Path(model_directory) / "hi.npy"),
            jij=np.load(Path(model_directory) / "Jij.npy"),
        )
        return instance


class AugmentedPotts:
    def __init__(
        self,
        potts_path: Union[Path, str] = None,
        msa_path: Union[Path, str] = None,
        encoder: str = "energies",
        alpha: float = 0.1,
    ) -> None:
        self.potts_path = potts_path
        self.msa_path = msa_path
        self.encoder = encoder
        self.alpha = alpha
        if msa_path is not None:
            logger.info("Running Gremlin locally and saving emission parameters")
            self.potts_model = PottsModel()
            self.potts_model.run_gremlin(msa_path=self.msa_path)
        elif self.potts_path is not None:
            logger.info(f"Loading Potts model locally from: {self.potts_path}")
            self.potts_model = PottsModel.load(self.potts_path)
        self.top_model = Ridge(alpha=self.alpha)

    def fit(self, data: pd.DataFrame, target: str) -> None:
        encodings = self.encode(data["sequence"])
        self.top_model.fit(encodings, data[target])

    def encode(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        return {
            "energies": self.compute_residue_energies,
            "augmented": self.compute_augmentation,
        }[self.encoder](sequences)

    def compute_residue_energies(
        self, sequences: Union[List[str], pd.Series]
    ) -> np.ndarray:
        embeddings = self.potts_model.embed(sequences)
        residue_energies = np.sum(embeddings, axis=2)
        return residue_energies

    def compute_augmentation(
        self, sequences: Union[List[str], pd.Series]
    ) -> np.ndarray:
        sequence_densities = self.potts_model.predict(sequences)
        one_hot_encodings = encode_sequences_one_hot(sequences).reshape(
            len(sequences), -1
        )
        return np.hstack((one_hot_encodings, sequence_densities.reshape(-1, 1)))

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        encodings = self.encode(data["sequence"])
        return self.top_model.predict(encodings)
