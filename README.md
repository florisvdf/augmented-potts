# Augmented Potts
The Augmented Potts supervised regression model for variant effect prediction as first described in: 
[Learning protein fitness models from evolutionary and assay-labeled data](https://doi.org/10.1038/s41587-021-01146-5) 
and exactly matching the implementation of the respective model used in: [Enzyme Structure Correlates With Variant Effect Predictability](https://doi.org/10.1016/j.csbj.2024.09.007)


# Installation

`augmented-potts` calls [`GMRELIN_CPP`](https://github.com/sokrypton/GREMLIN_CPP) to fit 
the Markov random field to the MSA and generate emission parameters used to compute 
sequence energies. Make sure that `GREMLIN_CPP` is installed and added to `$PATH`.

### Using pip

```
git clone https://github.com/florisvdf/augmented-potts.git
cd augmented-potts
pip install .
```

### Using uv
```
git clone https://github.com/florisvdf/augmented-potts.git
cd augmented-potts
uv sync
```

# Usage

Train the Augmented Potts model directly with the entrypoint script:

```
>>> uv run python src/augmented_potts/run.py --help
Usage: run.py [OPTIONS] DATA_PATH TARGET

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path      TEXT  [default: None] [required]                                                                                             │
│ *    target         TEXT  [default: None] [required]                                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --potts-path        TEXT   [default: None]                                                                                                       │
│ --msa-path          TEXT   [default: None]                                                                                                       │
│ --encoder           TEXT   [default: energies]                                                                                                   │
│ --alpha             FLOAT  [default: 0.1]                                                                                                        │
│ --output-dir        TEXT   [default: None]                                                                                                       │
│ --help                     Show this message and exit.                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Or import `AugmentedPotts` directly and train the model in your own script:

```
from augmented_potts.model import AugmentedPotts


model = AugmentedPotts()
model.fit(data=my_dataframe, target=my_target)
```

The `fit` method of the `AugmentedPotts` expects a dataframe with a `sequence` column 
storing protein variant sequences of identical length and will train on values in the 
column matching the string passed to the `target` argument.