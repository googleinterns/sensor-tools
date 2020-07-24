import ML_experiments as mle
import numpy as np
import pytest
import pytest_diff


def test_create_train_test_validation():
    MYDATA = {"pos": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
              "neg": [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]}
    actual_train, actual_test, actual_validation = mle.create_train_test_validation(
        MYDATA, "pos", "neg", 0.6, 0.2, 0.2)

    actual_train_pos = np.array(actual_train["pos"])
    actual_train_neg = np.array(actual_train["neg"])
    actual_test_pos = np.array(actual_test["pos"])
    actual_test_neg = np.array(actual_test["neg"])
    actual_validation_pos = np.array(actual_validation["pos"])
    actual_validation_neg = np.array(actual_validation["neg"])

    expected_train_pos = np.array([[0], [1], [2], [3], [4], [5]])
    expected_train_neg = np.array([[10], [20], [30], [40], [50], [60]])
    expected_test_pos = np.array([[6], [7]])
    expected_test_neg = np.array([[70], [80]])
    expected_validation_pos = np.array([[8], [9]])
    expected_validation_neg = np.array([[90], [100]])
    assert (expected_train_pos == actual_train_pos).all()
    assert (expected_train_neg == actual_train_neg).all()
    assert (expected_test_pos == actual_test_pos).all()
    assert (expected_test_neg == actual_test_neg).all()
    assert (expected_validation_pos == actual_validation_pos).all()
    assert (expected_validation_neg == actual_validation_neg).all()
