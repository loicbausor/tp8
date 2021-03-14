# Package : customMLLib
Very simple template for packaging ML project

## Content 
The package is seprated into two modules.

### Data.py 

This module (composed of the only class `DataManager`) allows to generates two interleaving half circles and split the dataset into validation and training set. The user can choose the number of data generated, the noise added to it and the proportion of the dataset to include in the validation set. 
In addition, the method "plot" allows to plot those data.

### Algorithms.py 

This module is composed of two classes.

##### TfLogisticRegression

Creates the logistic regression estimator to fit to a dataset. Fit this logistic regression on random batches. The user can of course choose the learning rate of the estimator : 'lr'
Three main interresting results can be called on a specific batch :
* Get the probability estimation 
* Get the class estimation 
* Get the accuracy of the prediction 

##### TfPerceptron 

Creates a two layers perceptron estimator to fit to the dataset. Fit the perceptron on random batches. The user can again choose the learning rate : 'lr', but also the number of desired neuron in the network : 'width'. 
Again, three main interresting results can be called on a specific batch :
* Get the probability estimation 
* Get the class estimation 
* Get the accuracy of the prediction

## Installation

```shell
pip install -e path/to/folder/customMLLib
```

## Example of usage

```python
from customML.Algorithms import TfLogisticRegression
from customML.Data import DataManager

# Creates the data
fraction = 0.1 # proportion of the validation set
n_data = 1000 # Size of the data 
noise = 0.1 # Noise in the data
data = DataManager(n_data=n_data, noise=noise, proportion=fraction)

# Gets training and validation sample
X_train, Y_train, X_val, Y_val = data.X_train, data.Y_train, data.X_val, data.Y_val

# Plots the data
data.plot() 

# Classification algorithms
# Instentiates Logistic Regression (same procedure for TfPerceptron)
logistic = TfLogisticRegression(lr=0.01)

# Fit the regression to the data
logistic.fit(X_train, Y_train)

# Gets the score on the validation data
logistic.score(X_val, Y_val)
```

# THE SCRIPT myscript.py

## Goal 
The aim of the script is to fit the two models we packaged on the data we also packaged (with custom inputs parameters) and to create a really basic HTML report of their performances. It includes the models & data parameters chosen, the models accuracies and some plots of the performances.

## Requirements
Appart from our package, don't forget to install the other requirements by the following command : 
```
pip install -r requirements
```
## Inputs 

The inputs of the scripts have to be specified in a YAML file. Its file path will be passed into the arguments `--input_yml` of the script. This file must contain the following fields (you can of course change the values): 

```{yaml}
# Toy data characteristics
n_data: 1000 # Number of data points
noise: 0.1 # Noise in the data

# Algorithms characteristics
lr: 0.01 # Positive
batch_size: 32 # Positive int > n_data
epochs: 100 # Positive int
width: 30 # The width of the perceptron : Postive int

# Report directory and name
report_directory: "" # A path 
# (if leaved empty it will create a report folder at the root repo)
report_name: "" # "Name.html" 
# (if leaved empty the report will be named Toy_report.html)
```
You will find this example YAML file in this repo (example_inputs.yaml).

## Execution

To run the code simply execute the myscript.py module with the YAML path needed: 

```
python -m myscript --input_yml Path/to/your/yml.yaml
```
