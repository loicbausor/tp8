# Package : customMLLib
Very simple template for packaging ML project

## The template
You will see that this package already has a setup.py, just provide your custom information, package requirements !

You already have an empty package folder with an ```__init__.py``` file inside ;-)

Just add some modules to it (.py files with functions and classes).

## Install
```pip install path/to/package/folder```

## Usage
```python
from customML.mymodule import myfunction
```

# Script : 

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
