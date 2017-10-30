# Project1
Project 1 for the EPFL ML course: https://mlo.epfl.ch/page-146520.html

## Structure
* **data** train.csv and test.csv must be put to this folder
* **src/run.py** is a file 
* **src** contains jupyter notebooks
* **src/scripts** contains python scripts (s.t. `implementations.py`)
* **src/run_tests.py** tests critical functions

## Installation
This project requires Python3 and virtualenv. Tested on Ubuntu Server 16.04.3 LTS
```
 $ sudo apt-get install python3-tk virtualenv python3
 $ ls README.md
# File should be in place
 $ virtualenv -p $(which python3) venv
 $ source venv/bin/activate
 $ pip install -r requirements.txt
 $ ls data/test.csv data/train.csv
# Files should be in place
 $ cd src
 $ python run_tests.py
# Tests should pass
 $ python run.py
# File logreg_1_submission.csv should be generated after ~10min
```

## What the run.py does
1. Loading data
2. Imputing missing features with mean values (taken w/o these missing values)
3. Creating 'missing feature' feature
4. Adding polynomial basis of degree 5
4. Replacing categorical features with several binary ones
5. Standardizing data (mean&std from train are used for test)
6. Training Logistic Regression with lambda=1e-4 using Netwon's method gamma=1e-1, 80 iterations, fixed random seed
7. Saving results to logreg_1_submission.csv
