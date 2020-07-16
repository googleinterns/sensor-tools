import windowing as wl
import numpy as np
import pytest


SAMPLE_MATRIX = {'x': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'y': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'z': [
    1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'nanos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
WINDOWED_DATA = [[[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, ], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]], [
    [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]]
NP_WINDOWED_DATA = np.array(WINDOWED_DATA, 'd')


def test_create_windows():

    window_size = 5
    stride = 5

    test_windowed_data = wl.create_windows_from_dictmatrix(
        SAMPLE_MATRIX, 4, window_size, stride)
    assert (test_windowed_data == NP_WINDOWED_DATA).all()


LIFT_WINDOWS = [[[20, 20, 1, 1, 1], [20, 20, 1, 1, 1],
                 [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]


def test_find_intial_lift_times():
    expected_start = 6
    expected_end = 10
    nplift = np.array(LIFT_WINDOWS)
    start, end = wl.find_initial_lift_times(nplift, 4)
    assert start == expected_start
    assert end == expected_end


def test_find_lift_windows():
    divisor = 4
    threshold = 0
    return_lift_windows = True
    l_windows = wl.find_lift_windows(
        NP_WINDOWED_DATA, divisor, threshold, return_lift_windows)
    actual_lift_windows = np.array(l_windows, 'd')
    np_lift_windows = np.array(LIFT_WINDOWS, 'd')
    assert (np_lift_windows == actual_lift_windows).all()


def test_refined_start_time():
    WINDOWED_DATA_REFINED = [[[[1, 1], [1, 1], [1, 1], [1, 2]], [[1, 5], [1, 5], [1, 10], [3, 4]], [[1, 1], [
        1, 1], [1, 1], [5, 6]], [[1, 20], [1, 20], [1, 20], [7, 8]], [[20, 20], [20, 20], [20, 20], [9, 10]]]]

    NP_WINDOWED_DATA_REFINED = np.array(WINDOWED_DATA_REFINED)
    threshold = 2
    divisor = 0
    return_lift_windows = False
    expected_start_time = 3
    actual_start_time = wl.find_lift_windows(
        NP_WINDOWED_DATA_REFINED, divisor, threshold, return_lift_windows)
    assert actual_start_time == expected_start_time


def test_find_precise_start_time():
    FRONT_WINDOW_LIFT = [[[1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 1, 2, 2, 2, 2, 20, 20, 20], [
        1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9]]]
    np_front_window_lift = np.array(FRONT_WINDOW_LIFT, 'd')
    rows = 4
    window_size = 2
    stride = 1
    threshold = 5
    expected_start_time = 6
    actual_start_time = wl.find_precise_start_time(
        np_front_window_lift, rows, window_size, stride, threshold)
    assert expected_start_time == actual_start_time


def test_place_data():
    LIFT = [[10, 20], [10, 20], [1, 2]]
    np_LIFT = np.array(LIFT)
    expected_train = {"data": [[[10], [10], [1]],
                               [[20], [20], [2]]], "labels": [1, 1]}
    train = {"data": [], "labels": []}
    data_key = "data"
    data_label = "labels"
    dictionary = train
    window_size = 1
    stride = 1
    positive = True
    actual_train = wl.place_data(
        np_LIFT, data_key, data_label, dictionary, positive, window_size, stride)
    np_expected_data = np.array(expected_train["data"])
    assert (expected_train["labels"] == actual_train["labels"])
    assert (np_expected_data == actual_train["data"]).all()


def test_choose_dataset():
    data_count = 5
    num_test = 3
    num_train = 5
    num_validation = 2
    expected_dataset = "train"
    actual_dataset = wl.choose_dataset(
        data_count, num_test, num_train, num_validation)
    assert expected_dataset == actual_dataset
    data_count = 0
    expected_dataset = "test"
    actual_dataset = wl.choose_dataset(
        data_count, num_test, num_train, num_validation)
    assert expected_dataset == actual_dataset
    data_count = 9
    expected_dataset = "validation"
    actual_dataset = wl.choose_dataset(
        data_count, num_test, num_train, num_validation)
    assert expected_dataset == actual_dataset

# def get_start_time(lift_windows, rows, window_size, stride, variance_threshold):


def test_get_start_time():
    FRONT_WINDOW_LIFT = [[[1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [
        1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]
    np_FRONT_WINDOW_LIFT = np.array(FRONT_WINDOW_LIFT)
    expected_start = 6
    expected_end = 10
    rows = 4
    window_size = 2
    stride = 1
    variance_threshold = 6
    actual_start, actual_end = wl.get_start_time(
        np_FRONT_WINDOW_LIFT, rows, window_size, stride, variance_threshold)
    assert expected_start == actual_start
    assert expected_end == actual_end
