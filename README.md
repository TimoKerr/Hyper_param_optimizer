[![Python application test with Github Actions](https://github.com/TimoKerr/Flask_app/actions/workflows/main.yml/badge.svg)](https://github.com/TimoKerr/Flask_app/actions/workflows/main.yml)

# Flask_app
Simple ML flask app to serve as template

# Make venv 
python3 -m venv ~/.Flask_ML_app
source ~/.Flask_ML_app/bin/activate

## Dir structure
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
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── util.cpython-38.pyc
│   └── util.py
├── __pycache__
│   ├── app.cpython-38-pytest-6.2.2.pyc
│   ├── app_test.cpython-38-pytest-6.2.2.pyc
│   └── iris.cpython-38-pytest-6.2.2.pyc
├── README.md
├── requirements.txt
├── static
└── templates
    ├── after.html
    └── home.html
```
    
## Functions
Makes very simple linear regression model form fixed data, saves model as pkl, runs flask app in local.

## CI
To have CI with Github actions we have the mail.yml file in .githun/workflows and the Makefil lint line.
