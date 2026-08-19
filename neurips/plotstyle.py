"""Shared paths and plotting style for the NeurIPS figures.

Every notebook in this folder imports from here, so fonts, ticks and frame
widths match the redshift-binned stellar-mass figure across all panels.
"""
import os
from pathlib import Path

import matplotlib as mpl

# The repository holds the code; the data and the training outputs live beside
# it in this same checkout. ROOT is taken from this file, so it is correct
# wherever the repository is cloned.
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "neurips"
PAPER_DIR = ROOT / "paper/ssl_outthere_paper"

DATA = ROOT / "data"
IMAGE_ROOT = DATA / "image"
DJA_FITS = DATA / "spectrum/DJA_spectra_v4.5.fits"
XMATCH = DATA / "crossmatched/dja_x_f150w.fits"
EMBED_H5 = DATA / "crossmatched/embeddings_f150w.h5"
COSMOS_PHOTOM = DATA / "survey/cosmos_2025/COSMOSWeb_mastercatalog_v1_photom_primary.fits"

IMAGE_DIR = ROOT / "encoder_image/jwst_dino"
SPEC_DIR = ROOT / "encoder_spectrum/LowResPT"
FUSION_DIR = ROOT / "encoder_fusion"

IMG_OUT = IMAGE_DIR / "outputs"
SPEC_OUT = SPEC_DIR / "outputs"
FUSION_OUT = FUSION_DIR / "outputs"

# Image-encoder checkpoint. This is the run the paper used; set it to your own
# run directory after pre-training.
IMG_CKPT = IMG_OUT / "jwst_dino_ps6_st3/version_6/checkpoints/last.ckpt"

PIX_SCALE = 0.030  # arcsec per pixel of the common F150W grid

RC = {
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
}

SPINE_LW, TICK_W, TICK_L = 1.8, 1.6, 4.2

# Per-modality colours, reused across figures.
C_IMAGE = "#006bff"
C_SPECTRUM = "#c76827"
C_ORIG = "#808080"
C_RED = "#8f2d34"         # the red of the model-overview figure
C_RED_BRIGHT = "#ff4136"  # readable on the dark grayscale cutouts
C_THIRD = "seagreen"
C_KNN = "darkorange"


def use_style(scale=1.0):
    """Apply the shared rcParams, with every font size multiplied by `scale`.

    A figure is placed in the paper at \linewidth, so a wide figure is shrunk
    more than a narrow one and the same point size ends up visually smaller.
    Each notebook passes the scale that brings its text close to the body font:
    roughly 5.5 / figure_width_in_inches, normalised so that 14 pt lands near
    9 pt on the page.
    """
    rc = {k: (v * scale if k.endswith("size") else v) for k, v in RC.items()}
    mpl.rcParams.update(rc)
    globals()["_SCALE"] = scale


def fs(size):
    """Scale an explicit font size by whatever `use_style` was called with."""
    return size * globals().get("_SCALE", 1.0)


def style_axes(ax, labelsize=None, **kwargs):
    """Thick frame and inward ticks on all four sides."""
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
    ax.tick_params(which="both", direction="in", width=TICK_W, length=TICK_L,
                   labelsize=labelsize if labelsize is not None
                   else mpl.rcParams["xtick.labelsize"],
                   top=True, bottom=True, left=True, right=True, **kwargs)


def scalebar(ax, n_pix=20, pix_scale=PIX_SCALE, color="w", loc=(0.06, 0.08), lw=2.5,
             fontsize=10):
    """Horizontal scale bar in the lower-left of an image panel, labelled in arcsec.

    Coordinates are axes fractions, so the bar length is converted from pixels
    through the panel's current x range.
    """
    x0, x1 = ax.get_xlim()
    frac = n_pix / abs(x1 - x0)
    xs, ys = loc
    ax.plot([xs, xs + frac], [ys, ys], transform=ax.transAxes, color=color, lw=lw,
            solid_capstyle="butt")
    ax.text(xs + frac / 2, ys + 0.025, f'{n_pix * pix_scale:.2g}"', transform=ax.transAxes,
            color=color, fontsize=fontsize, ha="center", va="bottom", fontweight="bold")


def save(fig, name, paper_name=None):
    """Write next to the notebooks and overwrite the copy used by the paper."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / f"{paper_name or name}.png", dpi=300, bbox_inches="tight")
