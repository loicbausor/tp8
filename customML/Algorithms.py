import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class TfLogisticRegression:
    """
    Creates a logistic regression estimator to fit to a dataset
    """
    def __init__(self, lr=0.01):
        """
        Instentiates the class 
        args:
            lr(float) : learning rate of our estimator
        """
        
        weight_init = tf.random_normal_initializer(mean=0., stddev=1.)
        self._w = tf.Variable(weight_init(shape=(2,1), dtype=tf.float32))
        self._b = tf.Variable(weight_init(shape=(), dtype=tf.float32))
        self.lr = lr
        
    def _compute_loss(self, e, y):
        """
        Computes the loss (binary crossentropy) of the model
        args:
            e (tf.Variable): The estimations of our batches
            y (np.array): The ground truth of those batches
        returns: 
            _loss(tf.constant): The loss on those batches
        """
        
        _loss = tf.keras.losses.binary_crossentropy(y, e)
        return(_loss) 
    
    def _get_random_batch(self, X_train, Y_train, batch_size):
        """
        Gets random batch for fitting
        args:
            X_train(np.array):  Regressors
            y_train(np.array):  Ground truth data
            batch_size(int): The size of the batch
        returns:
            X_batch, Y_batch (np.arrays): The batch to fit
        """
            
        if batch_size != int(batch_size) or batch_size <= 0:
            raise ValueError("Batch size is not a positive integer, don't be an idiot.")
        
        if batch_size > X_train.shape[0]:
            raise ValueError("Batch size is higher than the training size (%.0f > %.0f). \
                             Please reduce batch size or get more training data." 
                             % (batch_size, X_train.shape[0]))
     
        
        rnd_indices = np.random.randint(0, len(X_train), batch_size)
        X_batch = X_train[rnd_indices]
        Y_batch = Y_train[rnd_indices]
        return X_batch, Y_batch
    
    def fit_on_batches(self, X_train, Y_train,
                       n_epochs=10, batch_size=2**6,
                       plot_training=False, verbose=1):
        """
        Fit the logistic regression
        args:
            X_train, Y_train (np.arrays): The data input (regressors and labels)
            n_epochs (int): number of epochs 
            batch_size(int): Batch size (positive integer)
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations
        """
        if n_epochs != int(n_epochs) or n_epochs <= 0:
            raise ValueError("n_epochs is not a positive integer, don't be an idiot.")
        
        if verbose==1:
            print("Start the fit on the estimator.")
            
        trainProcess = tf.keras.optimizers.SGD(self.lr)
        self._loss_record = []
        for epoch in range(n_epochs):
          # draw a random batch to fit on
            X, Y = self._get_random_batch(X_train, Y_train, batch_size)
            with tf.GradientTape() as tape:
                # Model equation (use '@' operator)
                _estimation = self.estimate(X)
                # Compute error vector (mean squared error)
                _error = self._compute_loss(_estimation, Y)
            # record the loss
            self._loss_record.append(_error.numpy().mean())
            # get the gradients of your model
            _grads = tape.gradient(_error, [self._w, self._b])
            # update your model by applying computed gradients
            trainProcess.apply_gradients(zip(_grads, [self._w, self._b]))
        
        self._train_accuracy = self.score(X_train, Y_train)
        if verbose==1:
            print("Fitting fisnished succesfully.")
            print("Acuracy on the training set %.2f:" % self._train_accuracy)
         
        if plot_training:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self._loss_record)
            ax.set_title("Loss during the training phase")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss value")
            ax.grid()
            return fig
    
    def estimate(self, X):
        """
        Compute the estimation (probabilty) of the logistic regression for
        a specific batch
        args:
            X(np.array): The batch input
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations
        """
        
        _estimation = tf.math.sigmoid(X @ self._w + self._b)
        return _estimation
    
    def estimate_class(self, X):
        """
        Compute the estimation (class) of the logistic regression for
        a specific batch
        args:
            X(np.array): The regressors
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations (classes)
        """
        return tf.where(self.estimate(X) > 0.5, 1,0)
    
    def score(self, X, y):
        """
        Compute the accuracy on the on a specific batch
        args:
            X, y (np.arrays): Regressors and labels
        returns
            acc (float): The accuracy on this batch
        """
        acc = np.sum(self.estimate_class(X) == y) / y.shape[0]
        return acc
class TfPerceptron:
    """
    Creates a two layers perceptron estimator to fit to a dataset
    """

    def __init__(self, width=40, lr=0.01):
        """
        Instentiates the class 
        args:
            width (int): Positive number specifying the number of neurons of the perceptron
            lr (float): learning rate of our estimator
        """
        if width != int(width) or width <= 0:
            raise ValueError("Width is not a positive integer, don't be an idiot.")    
        
        
        weight_init = tf.random_normal_initializer(mean=0., stddev=1.)
        self._w = tf.Variable(weight_init(shape=(width,2), dtype=tf.float32))
        self._b = tf.Variable(weight_init(shape=(width, 1), dtype=tf.float32))
        self._w_p = tf.Variable(weight_init(shape=(1, width), dtype=tf.float32))
        self._b_p = tf.Variable(weight_init(shape=(1,1), dtype=tf.float32))
        
        self._width = width
        self.lr = lr
        
    def _compute_loss(self, e, y):
        """
        Computes the loss (binary crossentropy) of the model
        args:
            e (tf.Variable): The estimations of our batches
            y (np.array): The ground truth of those batches
        returns: 
            _loss(tf.constant): The loss on those batches
        """
        
        _loss = tf.keras.losses.binary_crossentropy(y, e)
        return(_loss) 
    
    def _get_random_batch(self, X_train, Y_train, batch_size):
        """
        Gets random batch for fitting
        args:
            X_train(np.array):  Regressors
            y_train(np.array):  Ground truth data
            batch_size(int): The size of the batch
        returns:
            X_batch, Y_batch (np.arrays): The batch to fit
        """
            
        if batch_size != int(batch_size) or batch_size <= 0:
            raise ValueError("Batch size is not a positive integer, don't be an idiot.")
        
        if batch_size > X_train.shape[0]:
            raise ValueError("Batch size is higher than the training size (%.0f > %.0f). \
                             Please reduce batch size or get more training data." 
                             % (batch_size, X_train.shape[0]))
     
        
        rnd_indices = np.random.randint(0, len(X_train), batch_size)
        X_batch = X_train[rnd_indices]
        Y_batch = Y_train[rnd_indices]
        return X_batch, Y_batch
    
    def fit_on_batches(self, X_train, Y_train,
                       n_epochs=10, batch_size=2**6,
                       plot_training=False, verbose=1):
        """
        Fit the Percertron
        args:
            X_train, Y_train (np.arrays): The data input (regressors and labels)
            n_epochs (int): number of epochs 
            batch_size(int): Batch size (positive integer)
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations
        """
        if n_epochs != int(n_epochs) or n_epochs <= 0:
            raise ValueError("n_epochs is not a positive integer, don't be an idiot.")
        
        if verbose==1:
            print("Start the fit on the estimator.")
            
        trainProcess = tf.keras.optimizers.SGD(self.lr)
        self._loss_record = []
        for epoch in range(n_epochs):
          # draw a random batch to fit on
            X, Y = self._get_random_batch(X_train, Y_train, batch_size)
            with tf.GradientTape() as tape:
                # Model equation (use '@' operator)
                _estimation = self.estimate(X)
                # Compute error vector (mean squared error)
                _error = self._compute_loss(_estimation, Y)
            # record the loss
            self._loss_record.append(_error.numpy().mean())
            # get the gradients of your model
            _grads = tape.gradient(_error, [self._w, self._b])
            # update your model by applying computed gradients
            trainProcess.apply_gradients(zip(_grads, [self._w, self._b]))
        
        self._train_accuracy = self.score(X_train, Y_train)
        if verbose==1:
            print("Fitting fisnished succesfully.")
            print("Acuracy on the training set %.2f:" % self._train_accuracy)
         
        if plot_training:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self._loss_record)
            ax.set_title("Loss during the training phase")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss value")
            ax.grid()
            return fig
    
    def estimate(self, X):
        """
        Compute the estimation (probabilty) of the Peceptron for
        a specific batch
        args:
            X(np.array): The batch input
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations
        """
        _l1 = self._w @ X.T + self._b
        _a1 = tf.nn.relu(_l1)
        _l2 = self._w_p @ _a1 + self._b_p
        _estimation = tf.transpose(tf.math.sigmoid(_l2))
        return _estimation
    
    def estimate_class(self, X):
        """
        Compute the estimation (class) of the logistic regression for
        a specific batch
        args:
            X(np.array): The regressors
        returns
            _estimatation (tf.constant) : the estimation for 
                each observations (classes)
        """
        return tf.where(self.estimate(X) > 0.5, 1,0)
    
    def score(self, X, y):
        """
        Compute the accuracy on the on a specific batch
        args:
            X, y (np.arrays): Regressors and labels
        returns
            acc (float): The accuracy on this batch
        """
        acc = np.sum(self.estimate_class(X) == y) / y.shape[0]
        return acc