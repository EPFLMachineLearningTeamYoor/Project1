# Imports
from scripts import proj1_helpers, helpers, implementation, feature_processing, k_fold
import numpy as np

# Configuration
train_path = '../data/train.csv'
test_path  = '../data/test.csv'
output_path = 'logreg_1_submission.csv'
deg = 5
lambda_ = 1e-4
gamma = 0.1

# Loading data
print("Loading data")
y, X, idx = proj1_helpers.load_csv_data(train_path)
y_t, X_t, ids_t = proj1_helpers.load_csv_data(test_path)

print("Preprocessing data")
X_p, (x_mean, x_std) = feature_processing.process_X(X, deg)
X_t_p, _ = feature_processing.process_X(X_t, deg, (x_mean, x_std))

#Logistic
y_01 = helpers.y_to_01(np.array([y]).T)
w0 = np.zeros((X_p.shape[1], 1))

print("Training 1")
np.random.seed(42)
w01, l = implementation.reg_logistic_regression_newton(y_01, X_p, lambda_ = lambda_,
                                                      initial_w = w0, max_iters = 50, gamma = gamma,
                                                      debug = False)

print("Training 2")
np.random.seed(42)
w010, l = implementation.reg_logistic_regression_newton(y_01, X_p, lambda_ = lambda_,
                                                      initial_w = w01, max_iters = 20, gamma = gamma,
                                                      debug = False)
y_pred = proj1_helpers.predict_labels(w010, X_t_p)
proj1_helpers.create_csv_submission(ids_t, y_pred, output_path)
