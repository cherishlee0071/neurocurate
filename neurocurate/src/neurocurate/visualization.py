"""Headless, exportable RDM and top-disagreement figures."""

from pathlib import Path
import numpy as np
from .rdm import validate_rdm
from .metadata import validate_ids


def plot_rdms(rdms, ids, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = validate_ids(ids)
    if not rdms:
        raise ValueError("no RDMs to visualize")
    fig, axes = plt.subplots(1, len(rdms), figsize=(5 * len(rdms), 4), squeeze=False)
    try:
        for ax, (name, rdm) in zip(axes[0], rdms.items()):
            values = validate_rdm(rdm)
            if len(values) != len(ids):
                raise ValueError("RDM and image IDs mismatch")
            image = ax.imshow(values, interpolation="nearest", cmap="viridis", vmin=0)
            ax.set(title=name, xlabel="image (aligned order)", ylabel="image (aligned order)")
            if len(ids) <= 20:
                ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=6)
                ax.set_yticks(range(len(ids)), ids, fontsize=6)
            fig.colorbar(image, ax=ax, label="distance")
        fig.tight_layout()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)


def plot_top_disagreement(scores, image_paths, output, top_n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    top = scores.sort_values(["score", "image_id"], ascending=[False, True]).head(top_n)
    if len(top) == 0:
        raise ValueError("empty score table")
    fig = plt.figure(figsize=(max(8, len(top) * 1.5), 6))
    grid = fig.add_gridspec(2, len(top), height_ratios=[2, 1])
    try:
        ax = fig.add_subplot(grid[0, :])
        locations = np.arange(len(top))
        width = 0.25
        for offset, column, label in [(-width, "disagreement", "Brain–vision disagreement"),
                                     (0, "consistency", "Subject consistency"),
                                     (width, "score", "NeuroCurate score")]:
            ax.bar(locations + offset, top[column], width, label=label)
        ax.set_xticks(locations, top.image_id, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        for i, row in enumerate(top.itertuples()):
            image_ax = fig.add_subplot(grid[1, i])
            path = Path(image_paths[row.image_id])
            if path.is_file():
                with Image.open(path) as image:
                    image_ax.imshow(image.convert("RGB"))
            else:
                image_ax.text(0.5, 0.5, "image unavailable", ha="center", fontsize=7)
            image_ax.axis("off")
        fig.tight_layout()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
