# MLHub - Machine Learning Hub

My notes, experiments, models, etc. wrapped into a library. Mainly for personal use only.

## Table of contents

- [MLHub - Machine Learning Hub](#mlhub---machine-learning-hub)
    - [Table of contents](#table-of-contents)
    - [Notes](#notes)
        - [Setup](#setup)
            - [Sphinx Docs](#sphinx-docs)

## Notes

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

#### Sphinx Docs

[![Developer TheProjectsGuy][dev-shield]][dev-profile-link]

[dev-shield]: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
[dev-profile-link]: https://github.com/TheProjectsGuy