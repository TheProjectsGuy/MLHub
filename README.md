# MLHub - Machine Learning Hub

[![GitHub: TheProjectsGuy/MLHub](https://img.shields.io/badge/_-TheProjectsGuy%2FMLHub-grey?logo=GitHub&logoColor=white&labelColor=black
)](https://github.com/TheProjectsGuy/MLHub)

My notes, experiments, models, etc. wrapped into a library. Mainly for personal use only.

## Table of contents

- [MLHub - Machine Learning Hub](#mlhub---machine-learning-hub)
    - [Table of contents](#table-of-contents)
    - [Notes](#notes)
        - [Setup](#setup)
            - [Sphinx Docs](#sphinx-docs)

## Notes

- Project is currently not accepting contributions (alpha stage).
- Follow Zen of Python `python -c "import this"`
- Checkpoints: `.pt` is for training (dict is stored), `.pth` is the final model (state dict directly stored)

### Setup

Setup anaconda environment (use `conda` or `mamba`) using

```bash
conda create -n ml-hub python=3.9
bash ./env_setup.sh -d
```

This will install everything (including developer tools).
To install only the core requirements in your current conda environment, run

```bash
bash ./env_setup.sh $CONDA_DEFAULT_ENV
```

To add this repo as a package in a conda environment

```bash
# Install (add the folder to conda.pth)
conda develop ./src
# Verify if this worked (path should be present)
cat $CONDA_PREFIX/lib/python3.9/site-packages/conda.pth
# Remove this (after testing is over)
conda develop -u ./src
```

#### Sphinx Docs

[![Developer TheProjectsGuy][dev-shield]][dev-profile-link]

[dev-shield]: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
[dev-profile-link]: https://github.com/TheProjectsGuy
