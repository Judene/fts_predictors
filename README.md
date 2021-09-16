# fts_predictors
fts_predictors (_Financial Time Series Predictors_) contains standardized code for training ML models on financial time
series. Models can either be trained _once-off_, or in a _walk-forward fashion_. Models included:

- [x] Linear Regression
- [x] Multi-layered Perceptron
- [x] GRU Neural Network
- [x] ARIMA
- [ ] Support Vector Regression
- [ ] Transformer Network

Installation on Anaconda
-------------------------
After cloning the repository, create a virtual environment.
A virtual environment creates a sandbox that does not affect the system Python.
```
conda create -n yourenvname python=3.6 anaconda
source activate yourenvname
conda install -n yourenvname [package]
```

Installation on Ubuntu 20.04+
-------------------------
```
apt-get update && apt-get install -y \
    python3.6 python3-pip git-core openssh-client python3-virtualenv \
    libssl-dev wget zlib1g-dev python3.6-tk
git clone git@github.com:Judene/fts_predictors.git
cd fts_predictors
virtualenv ve -p /usr/bin/python3.6
./ve/bin/pip install -r requirements/cpu.txt
```

==============================
 
Example notebooks
-------------------
To see how the code is used, please inspect the following notebooks:

- [ ] TODO: Add Notebooks!


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data from third party sources.
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    ├── notebooks          <- Jupyter notebooks.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── scripts            <- Scripts to run various implementations of the models.
    ├── src                <- Base model classes and various utils.
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------
