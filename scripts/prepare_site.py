#!/usr/bin/env python3
"""Stage repository notes for MkDocs without moving their source files."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".site-content"
CONTENT_MARKDOWN = (
    "B200_LOCALITY_DOMAIN_PROBE.md",
    "CUDA_VMM_COMMUNICATION_LAYERS.md",
    "DEEPSEEK_V41_CED_CSA2_ARCHITECTURE.md",
    "DSA_256K_QPAIR_KV_OVERLAP_OBSERVATIONS.md",
    "FAST_HADAMARD_TRANSFORM_GPU_OPTIMIZATION.md",
    "FLASHINFER_CAKE_KDA_PREFILL_HIGHLIGHTS.md",
    "FLASHINFER_CAKE_KDA_SMALL_BH_K1_PARALLELISM.md",
    "FLASHINFER_PR_4262_CORE_OPTIMIZATIONS.md",
    "MOONEP_PUBLIC_API_VISUAL_GUIDE.md",
    "dsa-backward-formula-derivation-and-tensor-shapes.md",
    "fp8-fp4-weight-quantization.md",
)
SITE_FILES = ("index.md", "about.md")
SITE_DIRECTORIES = ("assets", "javascripts", "pictures", "stylesheets")


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()

    for filename in CONTENT_MARKDOWN:
        source = ROOT / filename
        shutil.copy2(source, STAGE / filename)

    for filename in SITE_FILES:
        source = ROOT / filename
        shutil.copy2(source, STAGE / filename)

    for dirname in SITE_DIRECTORIES:
        source = ROOT / dirname
        shutil.copytree(source, STAGE / dirname)


if __name__ == "__main__":
    main()
