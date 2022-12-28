import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA() # Model Class Call
    # Train a GDA classifier
    model.fit(x_train, y_train) # Model is fit on train dataset
    # Plot decision boundary on validation set
    x_v, y_v = util.load_dataset(valid_path, add_intercept=False) # Validate Dataset is loaded
    theta = np.concatenate(model.theta)
    util.plot(x_v, y_v, theta, save_path[0:len(save_path)-3] + "png")
    # Use np.savetxt to save outputs from validation set to save_path
    probability = model.predict(x_v) # Predication Probs
    np.savetxt(save_path, probability) # are Saved in Text File Given
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        s1, s2 = x.shape
        self.theta = [np.zeros(1), np.zeros([s2, 1])]
        phi = np.mean(y)
        meo_0 = np.mean(x[y == 0], axis=0).reshape(s2, 1)
        meo_1 = np.mean(x[y == 1], axis=0).reshape(s2, 1)
        meo = [meo_0, meo_1]
        mat = np.zeros([s2, s2])
        for i in range(s1):
            cal_m = x[i, :].reshape(s2, 1) - meo[int(y[i])]
            mat += np.matmul(cal_m, cal_m.T)
        sigma = mat / s1
        self.theta[0] = np.log(
            phi / (1 - phi)) + 0.5 * np.matmul(
                np.matmul(meo_0.T, np.linalg.inv(sigma)
            ),
            meo_0
            ) - 0.5 * np.matmul(
            np.matmul(
                meo_1.T, np.linalg.inv(sigma)
            ),
            meo_1
        )
        self.theta[1] = np.matmul(
            np.linalg.inv(sigma).T, (meo_1 - meo_0)
        )
        # *** END CODE HERE ***
    def sigmoid(self, z):
        """
        Implementation of the sigmoid activation in numpy
        
        Args:
        z: numpy array of any shape
        """
        
        return 1 / (1 + np.exp(-z))
        
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        s1, s2 = x.shape
        z = np.matmul(x, self.theta[1]) + (np.ones([s1, 1]) * self.theta[0])
        probability = self.sigmoid(z).reshape(s1,)
        
        return probability
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
