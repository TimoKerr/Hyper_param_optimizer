[![Python application test with Github Actions](https://github.com/TimoKerr/Flask_app/actions/workflows/main.yml/badge.svg)](https://github.com/TimoKerr/Flask_app/actions/workflows/main.yml)

# Hyper Parameter Optimization with different methods
In this repository I test different methods for hyperparameter optimization. As a test case, I try to optimize the parameters of the SKlearn RandomForest classifier. Hyperparams to optimize: 
- Max_depth
- n_estimators
- criterion

# Methods
1. Simple manual grid search
2. Random grid search
3. Skopt search (Baysian optimization)
4. HyperOpt search (Baysian optimization)
5. Optuna (new library, favourite)

# Make venv 
python3 -m venv ~/.hyper_optimization
source ~/.hyper_optimization/bin/activate

## Dir structure
The main.py loads the data and calls all of the different methods and prints their results. /n
The methods themselves are defined as functions in ./mylib/hypteroptimize.py
```
.
├── app.py
├── app_test.py
├── data
│   └── iris_csv.csv
├── iri.pkl
├── iris.py
├── Makefile
├── model.ipynb
├── mylib
│   ├── __init__.py
│   └── util.py
├── README.md
├── requirements.txt
├── static
└── templates
    ├── after.html
    └── home.html
```

## CI
Makefile, requirements.txt, and app_test.py are all for proper code formatting and CI.
