import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# The dataset is in fact not in CSV format in the UCI Machine Learning Repository,
# the attributes are instead separated by whitespace.
# We can load this easily using the pandas library.

dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# We can create Keras models and evaluate them with scikit-learn by using handy
# wrapper objects provided by the Keras library.
# This is desirable, because scikit-learn excels at evaluating models

# ============================ BASELINE ======================================
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    # No activation function is used for the output layer because it is a regression
    # problem and we are interested in predicting numerical values directly without transform
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# The Keras wrapper object for use in scikit-learn as a regression estimator is called KerasRegressor.
# We create an instance and pass it both the name of the function to create the
# neural network model as well as some parameters to pass along to the fit() function
# of the model later, such as the number of epochs and batch size.
# Both of these are set to sensible defaults.

seed = 7 # fix random seed for reproducibility
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

#use 10-fold cross validation to evaluate the model.
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# ============================ STANDARDIZED =================================

# Concern: the input attributes all vary in their scales because they measure different quantities
# Continuing on from the above baseline model, we can re-evaluate the same model
# using a standardized version of the input dataset.
# Def: Standardization (Z-score normalization)
# The most commonly used technique, which is calculated using the arithmetic mean
# and standard deviation of the given data. However, both mean and standard deviation
# are sensitive to outliers, and this technique does not guarantee a common numerical
# range for the normalized scores. Moreover, if the input scores are not Gaussian distributed,
# this technique does not retain the input distribution at the output.
# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# ============================== DEEPER ======================================
# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 7
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# =============================== WIDER =====================================
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
