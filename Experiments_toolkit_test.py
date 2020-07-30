import Experiments_toolkit as extk
import numpy as np
import pytest
import pytest_diff
import windowing as wl


def test_create_train_test_validation():
    
    POSITIVES = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    NEGATIVES = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])
    actual_train, actual_test, actual_validation = extk.create_train_test_validation(
        POSITIVES, NEGATIVES, 0.6, 0.2, 0.2)

    actual_train_pos = np.array(actual_train["positives"])
    actual_train_neg = np.array(actual_train["negatives"])
    actual_test_pos = np.array(actual_test["positives"])
    actual_test_neg = np.array(actual_test["negatives"])
    actual_validation_pos = np.array(actual_validation["positives"])
    actual_validation_neg = np.array(actual_validation["negatives"])

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


def test_proprocess_data():
    dict1 = {'x': [0, 0, 0, 0, 10, 20, 30, 10, 10, 10, 10], 'y': [0, 0, 0, 0, 10, 20, 30, 40, 10, 10, 10], 'z': [
        0, 0, 0, 0, 10, 20, 40, 1, 10, 10, 10], 'nanos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    dict2 = {'x': [0, 0, 0, 0, 10, 20, 30, 10, 10, 10, 10], 'y': [0, 0, 0, 0, 10, 20, 30, 40, 10, 10, 10], 'z': [
        0, 0, 0, 0, 10, 20, 40, 1, 10, 10, 10], 'nanos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

    DATA = [wl.dict_to_numpy_array(dict1), wl.dict_to_numpy_array(dict2)]
    rows = 4
    params1 = wl.Initial_Filtering_Params(3, 1, 10, 4)
    params2 = wl.Precise_Filtering_Params(2, 1, 1, 4)
    filtering_params = wl.All_Filtering_Params(params1, params2)
    window_size = 2
    offset = 0
    expected_dict = [[[[0, 10],
                      [0, 10],
                      [0, 10],
                      [4, 5]],  [[0, 10],
                                 [0, 10],
                                 [0, 10],
                                 [4, 5 ]]]]
    
    actual_dict = extk.preprocess_data(
        DATA, rows, filtering_params, wl.get_window_from_timestamp,window_size, offset)
    expected_windows = np.array(expected_dict)
    actual_windows = np.array(actual_dict)

    assert (expected_windows == actual_windows).all()


def test_get_np_X_Y():
    CAT1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    CAT2 = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    length = 5
    expected_Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    actual_X, actual_Y = extk.get_np_X_Y(CAT1, CAT2, length)
    assert (expected_Y == actual_Y).all()
    assert len(actual_X) == 2 * length
