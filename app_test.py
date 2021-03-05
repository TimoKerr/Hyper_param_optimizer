""" Just a test file"""
from mylib import util
import pickle
import numpy as np

# needs test_* function to run on pytest
def test_function():
    model = pickle.load(open("iri.pkl", "rb"))
    test_array = np.array([[1, 0.2, 0.3, 1]])
    pred = model.predict(test_array)
    success = util.training_complete()
    return success, pred
