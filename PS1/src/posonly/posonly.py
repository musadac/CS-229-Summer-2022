import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***

    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_t, y_t = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    model = LogisticRegression(verbose=False)
    model.fit(x_t, y_t)
    prob_test = model.predict(x_test)
    np.savetxt(output_path_true, prob_test)
    util.plot(x_test, y_test, model.theta, output_path_true[:-3] + "png")
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_t, y_t = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="y", add_intercept=True)
    model = LogisticRegression(verbose=False)
    model.fit(x_t, y_t)
    prob_test = model.predict(x_test)
    np.savetxt(output_path_naive, prob_test)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    util.plot(x_test, y_test, model.theta, output_path_naive[:-3] + "png")
    # Part (f): Apply correction factor using validation set and test on true labels
    x_test, y_test = util.load_dataset(test_path, label_col="y", add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col="y", add_intercept=True)
    model = LogisticRegression(verbose=False)
    model.fit(x_t, y_t)
    pred = model.predict(x_val)
    alpha = np.mean(pred[y_val == 1])
    prob_test = model.predict(x_test)
    pt_test = prob_test / alpha
    np.savetxt(output_path_adjusted, pt_test)
    x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
    util.plot(x_test, y_test, model.theta, output_path_adjusted[:-3]+"png",alpha)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
