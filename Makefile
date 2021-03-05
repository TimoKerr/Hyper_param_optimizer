setup:
	apt-get install python3-env
	python3 -m venv ~/.myrepo

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C app.py iris.py mylib/*.py

test: 
	python -m pytest -vv --cov=mylib --cov=app_test app_test.py

format:
	black *.py mylib/*.py