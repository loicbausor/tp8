# coding: utf8

# Logs and arguments
import os
import argparse
import logging
import yaml

# To make the script to work
from customML.Algorithms import TfPerceptron, TfLogisticRegression
from customML.Data import DataManager
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

# Custom Errors
class ArgsError(Error):
    pass

# Parser
parser = argparse.ArgumentParser()

parser.add_argument("--input_yml", type=str,
                    help="Path to the config yml file.")

args = parser.parse_args()

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("myscript.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# functions

def plot_frontier(estimator, X_val, Y_val):
    """
    Plot the classification frontier of a classifier
    """
    xx0 = np.linspace(-2.1,2.1,50)
    xx1 = np.linspace(-2.1,2.1,50)
    XX0, XX1 = np.meshgrid(xx0, xx1)
    
    XX01 = np.stack([XX0.reshape(-1), XX1.reshape(-1)], axis=1)

    hat_Y_proba = estimator.estimate(XX01).numpy()
    hat_Y_proba = hat_Y_proba.reshape(XX0.shape)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(hat_Y_proba, interpolation="nearest", origin="lower", cmap="jet", extent=[-2.1, 2.1, -2.1, 2.1])
    ax.scatter(X_val[:,0], X_val[:,1], c=Y_val[:,0], cmap="jet", marker=".", edgecolors="k")
    ax.set_title("Classification Frontier")
    return(fig)

# main function of your script
if __name__ == "__main__":
    logger.info("starting myscript :-)")
    try:
        # Args parsing & normalizing
        try:
            with open(args.input_yml, 'rb') as f:
                conf = yaml.safe_load(f.read())
        except:
            logger.critical("YAML conf file was not found.")
            raise ArgsError("The YAML to parse inputs was not found" +
            " (check if it exists and if it is in the good format).")    
        try:
            n_data = conf['n_data']
            noise = conf['noise']
            lr = conf['lr']
            batch_size = conf['batch_size']
            epochs = conf['epochs']
            width = conf['width']
            report_directory = conf['report_directory']
            report_name = conf['report_name']
        except:
            logger.critical("Some inputs are missing.")
            raise ArgsError("Some inputs are missing in the YAML configuration file." +
            "Check the documentation for more informations about the inputs configuration.")
        
        if report_name == "" or report_name is None:
            report_name = "Toy_report.html"
            logger.warning("No report name were given, it was " +
            "automatically assigned to : " + report_name)
        
        if report_name[-5:] != ".html":
            report_name = report_name + ".html"

        if report_directory == "" or report_directory is None:
            report_directory = os.getcwd()
            logger.warning("No report folder were given, it was " +
            "automatically assigned to : " + report_directory)
        
        report_directory = os.path.normpath(report_directory)
        
        # Make the report directory
        if not os.path.exists(report_directory + "/report"):
            os.mkdir(report_directory + "/report")
        
        logger.info("Arguments parsed succesfully.")

        plots = {}
        # Data managing
        data = DataManager(n_data=n_data, noise=noise)
        X_train, Y_train, X_val, Y_val = data.X_train, data.Y_train, data.X_val, data.Y_val
        plots['data_viz'] = data.plot()

        logger.info("Data loaded successfully.")

        # Algorithm fit & evaluation
        logitstic = TfLogisticRegression(lr = lr)
        plots['loss_1'] = logitstic.fit_on_batches(X_train, Y_train,
                                    plot_training = True, batch_size = batch_size,
                                    n_epochs = epochs, verbose=0)
        plots['frontier_1'] = plot_frontier(logitstic, X_val, Y_val)
        train_acc_1 = round(logitstic._train_accuracy,2)
        test_acc_1 = round(logitstic.score(X_val, Y_val),2)
        
        # Algorithm fit & evaluation
        perc = TfPerceptron(width=width, lr = lr)
        plots['loss_2'] = perc.fit_on_batches(X_train, Y_train,
                                    plot_training = True, batch_size = batch_size,
                                    n_epochs = epochs, verbose=0)
        plots['frontier_2'] = plot_frontier(perc, X_val, Y_val);  
        train_acc_2 = round(perc._train_accuracy,2)
        test_acc_2 = round(perc.score(X_val, Y_val),2)

        logger.info("Models fitted successfully.")

        # Transformation of the plot into png
        directory = report_directory + "/report/models_plot/"
        if not os.path.exists(directory):
            os.mkdir(directory)

        for name, plot in plots.items(): 
            plot.savefig(directory + name)
            plots[name] = "models_plot/" + name + ".png"

        
        # Creates the report
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("/templates/template.html")

        template_vars = {
            "lr": lr ,
            "n_data": n_data,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_acc_1": train_acc_1,
            "test_acc_1" : test_acc_1,
            "train_acc_2": train_acc_1,
            "test_acc_2" : test_acc_2,
            "width" : width,
            "data_viz": plots["data_viz"],
            "loss_1":plots["loss_1"],
            "frontier_1": plots["frontier_1"],
            "loss_2":plots["loss_2"],
            "frontier_2": plots["frontier_2"],
            "noise" : noise
            }
        
        html_out = template.render(template_vars)
        # to save the results
        with open(report_directory + "/report/" +report_name, "w") as fh:
            fh.write(html_out)
        
        logger.info("Code ended without errors, please find your report at : " +
        os.path.normpath(report_directory + "/report/" + report_name))
    except:
        logger.critical('Code ended with errors.')
