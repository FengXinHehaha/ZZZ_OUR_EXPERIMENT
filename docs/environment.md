# Environment Setup

## Conda Environment

- Environment name: `ZZZ_OUR_EXPERIMENT`
- Environment path: `/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT`
- Python version: `3.11.15`

## Activation

If `conda` is not already available in the shell:

```bash
source /home/fxh/anaconda3/etc/profile.d/conda.sh
conda activate ZZZ_OUR_EXPERIMENT
```

## Minimal Dependency Policy

This repository starts from a clean Python environment and installs only the packages needed by the rebuilt experiment pipeline.

Current baseline dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Notes

- Do not copy the old environments directly from previous projects.
- Add new dependencies only when they are required by the new pipeline.
- For heavyweight graph or deep learning packages such as `torch`, `torch-geometric`, `torch-scatter`, and `torch-sparse`, record exact versions here before installation so the setup stays reproducible.
