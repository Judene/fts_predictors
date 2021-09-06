# Judene Thesis
This repository contains all the code for the Master's thesis, Identifying generic machine learning approaches for financial time series forecasting. An abstract of the thesis follows:

A package for testing various time series predictors in a walk forward fashion

# Installation on Anaconda
After cloning the repository, create a virtual environment.
A virtual environment creates a sandbox that does not affect the system Python.
```
conda create -n yourenvname python=3.6 anaconda
source activate yourenvname
conda install -n yourenvname [package]
```
==============================
 
Example notebooks
-------------------
To see how the code is used, please inspect the following notebooks:

A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
