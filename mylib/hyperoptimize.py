""" Different hyperparameter optimisation methods"""
import numpy as np

# General
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from functools import partial

# Skopt
from skopt import space
from skopt import gp_minimize

# Hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

# Optuna
import optuna



def GridSearch():
    """ Does the basic hyperparam grid search"""
    print("Starting Grid Search")
    # Define Classifier
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators":[100,200],
        "max_depth":[1,3],
        "criterion": ["gini","entropy"]
    }

    # Define the gridsearch
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=0,
        n_jobs=1,
        cv=5
    )
    return model


def RandomSearch():
    """ Does random grid search"""
    print("Starting Random Grid Search")
    # Define Classifier
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators":np.arange(100,1500,100),
        "max_depth": np.arange(1,20),
        "criterion": ["gini","entropy"]
    }

    # Define the grid search
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter= 10,
        scoring="accuracy",
        verbose=0,
        n_jobs=1,
        cv=5
    )
    return model


def SkoptSearch(train_x, train_y):
    """Usining skopt library to do Baysian Optimization of hyperparameters.
    Returns a dictionary with best params."""
    print("Starting SkoptSearch.")
    # First define the optimize function
    def optimize(params, param_names, x, y):
        params = dict(zip(param_names, params))
        model = ensemble.RandomForestClassifier(**params)
        kf = model_selection.StratifiedKFold(n_splits=5)
        accuracies = []
        # split data
        for idx in kf.split(X= x, y=y):
            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            model.fit(xtrain, ytrain)
            preds = model.predict(xtest)
            fold_acc = metrics.accuracy_score(ytest, preds)
            accuracies.append(fold_acc)
        # Return the thing to minimize (-1 because lower is better)
        return -1.0 * np.mean(accuracies)

    #Define param space for skopt
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")  
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]

    optimization_function = partial(
        optimize,
        param_names=param_names,
        x = train_x,
        y = train_y
    )

    # run gpminimize functions
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )
    results = dict(zip(param_names,result.x))
    return results


def HyperOptSearch(train_x, train_y):
    """ Using HyperOpt library. Returns dictionary with best Hyper Params"""

    # everything from skopt remains the same, only difference
    # is that the input is now params as dict with names in.
    def optimize(params, x, y):
        model = ensemble.RandomForestClassifier(**params)
        kf = model_selection.StratifiedKFold(n_splits=5)
        accuracies = []
        # split data
        for idx in kf.split(X= x, y=y):
            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            model.fit(xtrain, ytrain)
            preds = model.predict(xtest)
            fold_acc = metrics.accuracy_score(ytest, preds)
            accuracies.append(fold_acc)
        # Return the thing to minimize (-1 because lower is better)
        return -1.0 * np.mean(accuracies)

    # Defining param spac enow requires defining a dict 
    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.01, 1) 
    }

    optimization_function = partial(
        optimize,
        x = train_x,
        y = train_y
    )
    
    trials = Trials()
    result = fmin(
        fn = optimization_function,
        space=param_space,
        algo = tpe.suggest,
        max_evals=15,
        trials = trials
    )

    return result


def OptunaSearch(train_x, train_y):

    def optimize(trial, x, y):
        trials = Trials()
        criterion = trial.suggest_categorical("criterion",["gini","entropy"])
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        max_features = trial.suggest_uniform("max_features", 0.01, 1.0)

        model = ensemble.RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            max_features = max_features,
            criterion = criterion
        )
        kf = model_selection.StratifiedKFold(n_splits=5)
        accuracies = []
        # split data
        for idx in kf.split(X= x, y=y):
            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            model.fit(xtrain, ytrain)
            preds = model.predict(xtest)
            fold_acc = metrics.accuracy_score(ytest, preds)
            accuracies.append(fold_acc)
        # Return the thing to minimize (-1 because lower is better)
        return -1.0 * np.mean(accuracies)

    optimization_function = partial(optimize, x=train_x, y=train_y)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)

    return study