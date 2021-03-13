from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

class DataManager: 
    
    def __init__(self, n_data=100, noise=None, proportion=None) :
        
        """
        Instantiates the class
        Generates two interleaving half circles and split the dataset into validation and training set. 
        
        Parameters
        ----------
        n_data : int or tuple of shape(2,), default = 100
                  If int, the total number of points generated 
                  If two-element tuple, number of points in each of the two moons
        noise : float, default = None 
                Standart deviation of gaussian noise added to the data 
                Must be non-negative
        proportion: float, default = None 
                    Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split
        """
        
        self.n_data = n_data 
        self.noise = noise 
        
        #Generates the 
        X, Y = make_moons(self.n_data, noise=self.noise, random_state=42)
        Y = Y[:,np.newaxis]

        self.X_train, self.Y_train, self.X_val, self.Y_val = self.split_dataset(X, Y, proportion)
        
        
        
    def split_dataset(self, X, Y, proportion=None): 
        
        """
        Split the dataset into a training and a vaidation set. 

        Parameters
        ----------
        X : array 
        Y : array 
        proportion: float, default = None 
                    Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split
                    If None, it will be set to 0.25
        Returns
        -------
        X_train : array of length Len(X)*(1-proportion)
        y_train : array of length Len(X)*(1-proportion) 
        X_val : array of length Len(X)*proportion
        Y_val : array of length Len(X)*proportion
        
        """
        
        if proportion == None:
            proportion = 0.25
        if proportion < 0.0 or proportion > 1.0 : 
            raise ValueError("Proportion must be between 0.0 and 1.0")
        val_size = int(len(X) * proportion)
        X_train = X[:-val_size]
        Y_train = Y[:-val_size]
        X_val = X[-val_size:]
        Y_val = Y[-val_size:]
        
        return X_train,Y_train, X_val, Y_val
    
        
    def plot(self): 
        
        """
        Plot the splitted data
        
        """

        fig, ax = plt.subplots(figsize=(10,10))

        blue_points_train = self.X_train[self.Y_train[:, 0] == 0]
        x_blue_train = blue_points_train[:, 0]
        y_blue_train = blue_points_train[:, 1]


        blue_points_valid = self.X_val[self.Y_val[:, 0] == 0]
        x_blue_valid = blue_points_valid[:, 0]
        y_blue_valid = blue_points_valid[:, 1]

        red_points_train = self.X_train[self.Y_train[:, 0] == 1]
        x_red_train = red_points_train[:, 0]
        y_red_train = red_points_train[:, 1]

        red_points_valid = self.X_val[self.Y_val[:, 0] == 1]
        x_red_valid = red_points_valid[:, 0]
        y_red_valid = red_points_valid[:, 1]

        ax.scatter(x_blue_train, y_blue_train,
           c="blue", marker=".",
           label="training class 'blue'")
        ax.scatter(x_blue_valid, y_blue_valid, 
           c="blue", marker="+", 
           alpha=0.3, label="validation class 'blue'")
        ax.scatter(x_red_train, y_red_train,
           c="red", marker=".", 
           label="training class 'red'")
        ax.scatter(x_red_valid, y_red_valid, 
           c="red", marker="+", 
           alpha=0.3, label="validation class 'red'")
        
        ax.grid()
        ax.set_title('Visualisation of the data')
        fig.legend()
        
        return fig