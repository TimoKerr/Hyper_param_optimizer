""" Just a test file"""
from mylib import util
print(util.training_complete())

# needs test_* function to run on pytest
def test_function():
    success = util.training_complete()
    return success