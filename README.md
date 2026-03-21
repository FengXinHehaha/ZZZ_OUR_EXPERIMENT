# ZZZ_OUR_EXPERIMENT

This repository is the clean starting point for our rebuilt APT experiment pipeline.

## Purpose

- Keep the new experiment line isolated from older versioned directories.
- Rebuild the workflow from scratch in a reproducible and Git-managed workspace.
- Migrate ideas selectively from older implementations instead of copying old code wholesale.

## Repository Scope

- Root path: `/home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT`
- Git only manages files inside this directory.
- Older directories such as `26.03.17 version`, `KAIROS_CODE`, and `OCRAPT_CODE` remain external references.

## Initial Layout

- `docs/` experiment notes, design decisions, and evaluation plans
- `src/` source code for the new pipeline
- `configs/` configuration files and environment settings
- `artifacts/` runtime outputs, logs, checkpoints, and generated data

## Working Principles

- Start from a minimal baseline and add components intentionally.
- Prefer reproducible scripts over ad hoc notebooks.
- Keep large data and generated artifacts out of Git.

## Next Steps

- Define the first baseline experiment.
- Decide data extraction and graph-building conventions.
- Establish training, detection, and evaluation entrypoints.
