# source: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("./pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define fully connected layers using the Dense class.
# first argument = number of neurons in the layer
# second argument = initialization method (default = uniform)
# "activation" argument = activation function
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# specify:  the loss function to use to evaluate a set of weights (logarithmic loss)
#           the optimizer used to search through different weights for the network (adam)
#           any optional metrics we would like to collect and report during training.

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)

#Now we can evaluate the model or make predictions with it

# Evaluate the model (passing the same i/o used to train the model)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
