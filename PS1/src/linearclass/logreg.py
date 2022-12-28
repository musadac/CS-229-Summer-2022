import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression() # Model Class Call
    # Train a logistic regression classifier
    model.fit(x_train, y_train) # Model is fit on train dataset
    # Plot decision boundary on top of validation set set
    x_v, y_v = util.load_dataset(valid_path, add_intercept=True) # Validate Dataset is loaded
    util.plot(x_v, y_v, model.theta, save_path[0:len(save_path)-3] + "png") # Given Plot Function is Used
    # Use np.savetxt to save predictions on eval set to save_path
    probaility = model.predict(x_v) # Predication Probs
    np.savetxt(save_path, probaility) # are Saved in Text File Given
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
    def sigmoid(self, z):
        """
        Implementation of the sigmoid activation in numpy
        
        Args:
        z: numpy array of any shape
        """
        
        return 1 / (1 + np.exp(-z))
    def hessian(self, x, y):
        """
        As Calculated in Part a
        """
        s1, s2 = x.shape # Get the shape of x
        h = np.zeros([s2, s2]) # Generate x by x rows of hessian
        for i in range(s2): # for x i
            for j in range(s2): # for x j
                h[i][j] = np.mean(self.sigmoid(np.matmul(x, self.theta)) * (1 - self.sigmoid(np.matmul(x, self.theta))) * x[:, i] * x[:, j]) # Calculating Hessian using the mean g(xthetha)*(1-g(xthetha))xixj
        return h

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        s1, s2 = x.shape # Get the shape of x
        self.theta = np.zeros(s2) # We will initialize theta to be zero
        step = 0 # Step Counter to see number of iters done
        delta = np.inf # Big Number to be replaced with lower
        while delta >= self.eps and step < self.max_iter: # Until Converges or Max Iter Exceeds
            n_f = np.zeros(s2) # Function F0 F1 F2 .... Fn Calculation Array Initialized with 0
            for i in range(s2): # For all F Calculate F
                n_f[i] = - np.mean((y - self.sigmoid(np.matmul(x, self.theta))) * x[:, i])
            h = self.hessian(x, y) # Get the Hessian
            new_the = self.theta - np.matmul(np.linalg.inv(h), n_f) # Multiplying Inverse of Hessian with Function Nabala
            delta = np.linalg.norm(new_the - self.theta) # Delta Values are Normalized and Saved
            self.theta = new_the # New Thetha
            if self.verbose: # Verbose Functionality to show the normal Parameter Changes at each iteration
                print(f"Epoch {step}: parameter change to: {delta}.")
            step += 1
        print(f"Converges after {step} epochs having delta: {delta}.")
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        prob = self.sigmoid(np.matmul(x, self.theta)) # Predict the Value 
        return prob
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
