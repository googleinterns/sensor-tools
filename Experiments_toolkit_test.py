import Experiments_toolkit as extk
import numpy as np
import tensorflow.data as tfdata
import pytest
import pytest_diff
import windowing as wl

POSITIVES = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
NEGATIVES = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])

def test_create_train_test_validation():
    actual_train, actual_test, actual_validation = extk.create_train_test_validation(
        POSITIVES, NEGATIVES, 0.6, 0.2, 0.2)

    actual_train_pos = len(np.array(actual_train["positives"]))
    actual_train_neg = len(np.array(actual_train["negatives"]))
    actual_test_pos = len(np.array(actual_test["positives"]))
    actual_test_neg = len(np.array(actual_test["negatives"]))
    actual_validation_pos = len(np.array(actual_validation["positives"]))
    actual_validation_neg = len(np.array(actual_validation["negatives"]))

    expected_train_pos = len(np.array([[0], [1], [2], [3], [4], [5]]))
    expected_train_neg = len(np.array([[10], [20], [30], [40], [50], [60]]))
    expected_test_pos = len(np.array([[6], [7]]))
    expected_test_neg = len(np.array([[70], [80]]))
    expected_validation_pos = len(np.array([[8], [9]]))
    expected_validation_neg = len(np.array([[90], [100]]))
    assert (expected_train_pos == actual_train_pos)
    assert (expected_train_neg == actual_train_neg)
    assert (expected_test_pos == actual_test_pos)
    assert (expected_test_neg == actual_test_neg)
    assert (expected_validation_pos == actual_validation_pos)
    assert (expected_validation_neg == actual_validation_neg)


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

def test_create_dataset():
    seed = 0
    pos = POSITIVES
    neg = NEGATIVES
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    label_pos = np.ones(len(POSITIVES))
    label_neg = np.zeros(len(NEGATIVES))
    labels = np.concatenate((label_pos, label_neg))
    examples = np.concatenate((POSITIVES, NEGATIVES))
    max_size = -1
    expected_dataset = tfdata.Dataset.from_tensor_slices((examples, labels))
    actual_dataset = extk.create_dataset(POSITIVES, NEGATIVES, max_size)
    expected_set = set(())
    for e in expected_dataset.as_numpy_iterator():
        pairing = str(e[0][0]) + ", " + str(e[1])
        expected_set.add(pairing)
    actual_set = set(())
    for a in actual_dataset.as_numpy_iterator():
        pairing = str(a[0][0]) + ", " + str(a[1])
        actual_set.add(pairing)
    assert actual_set == expected_set

def test_get_np_examples_labels():
    CAT1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    CAT2 = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    length = 5
    expected_Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    actual_X, actual_Y = extk.get_np_examples_labels(CAT1, CAT2, length)
    assert (expected_Y == actual_Y).all()
    assert len(actual_X) == 2 * length
