import pandas as pd
import typer
from loguru import logger
from scipy.stats import spearmanr

from augmented_potts.model import AugmentedPotts


def main(
    data_path: str,
    target: str,
    potts_path: str = None,
    msa_path: str = None,
    encoder: str = "energies",
    alpha: float = 0.1,
    output_dir: str = None,
):
    data = pd.read_csv(data_path)
    model = AugmentedPotts(
        potts_path=potts_path,
        msa_path=msa_path,
        encoder=encoder,
        alpha=alpha,
    )
    model.fit(data=data, target=target)
    spearman_scores = {}
    split = data["split"]
    predictions = pd.DataFrame(split)
    numeric_split = split.reset_index().rename(columns={0: "split"})
    y_pred = model.predict(data)
    for prefix, indices in numeric_split.groupby("split"):
        y_split_pred = y_pred[indices.index.values]
        predictions.loc[lambda d: d["split"] == prefix, "y_pred"] = y_split_pred
        spearman_scores[prefix] = spearmanr(
            y_split_pred, data.loc[indices.index.values][target].values
        ).correlation
    for prefix, spearman in spearman_scores.items():
        logger.info(f"{prefix} spearman correlction: {spearman:.3f}")
    if output_dir is not None:
        predictions.to_csv(output_dir / "predictions.csv", index=True)


if __name__ == "__main__":
    typer.run(main)
