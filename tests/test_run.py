import random
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest
from biotite.sequence.io.fasta import FastaFile

from augmented_potts.constants import AA_ALPHABET
from augmented_potts.run import main
from augmented_potts.utils import get_dummy_msa, write_fasta


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    n_samples = 100
    mut_prob = 0.2
    reference_sequence = "SEQWENCE"
    sequences = set()
    while len(sequences) < n_samples:
        sequences.add(
            "".join(
                [
                    random.choices(
                        [aa_ref, random.choice(AA_ALPHABET)], [1 - mut_prob, mut_prob]
                    )[0]
                    for aa_ref in reference_sequence
                ]
            )
        )

    return pd.DataFrame(
        dict(
            sequence=list(sequences),
            split=["train", "valid"] * int(n_samples / 2),
            a=np.random.normal(0, 1, n_samples),
            b=np.random.normal(1, 1.5, n_samples),
            c=[3, 4] * int(n_samples / 2),
            d=["a", "b"] * int(n_samples / 2),
        )
    )


@pytest.fixture(scope="session")
def tmpdir_session() -> Path:
    path = mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture(scope="session")
def dummy_msa(data: pd.DataFrame):
    sequence_length = len(data["sequence"].iloc[0])
    msa = get_dummy_msa(sequence_length=sequence_length)
    return msa


@pytest.fixture(scope="session")
def run_args(data: pd.DataFrame, dummy_msa: FastaFile, tmpdir_session: Path) -> dict:
    data_path = Path(tmpdir_session) / "data.csv"
    msa_path = Path(tmpdir_session) / "msa.a3m"
    data.to_csv(data_path)
    write_fasta(msa_path, dummy_msa)
    args = dict(
        data_path=str(data_path),
        target="a",
        potts_path=None,
        msa_path=str(msa_path),
        encoder="energies",
        alpha=0.1,
        output_dir=tmpdir_session,
    )
    return args


def test_run(run_args: dict) -> None:
    main(**run_args)


def test_run_stores_predictions(run_args: dict) -> None:
    predictions_path = Path(run_args["output_dir"]) / "predictions.csv"
    assert predictions_path.exists()
