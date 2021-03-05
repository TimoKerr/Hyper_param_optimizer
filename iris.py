import pandas as pd
import numpy as np
import pickle

# read data from local csv
df = pd.read_csv("./data/iris_csv.csv")
df.head()

# get out input and target
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

# one hot encode target
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# split data into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# load linear regression model and actually train
from sklearn.svm import SVC

sv = SVC(kernel="linear").fit(X_train, y_train)
# using the (superfluous) util function in mylib
from mylib import util

print(util.training_complete())

# save trained model as serialised pickle
pickle.dump(sv, open("iri.pkl", "wb"))
