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
    start, end = wl.find_initial_lift_times(LIFT_WINDOWS, 4)
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
    WINDOWED_DATA_REFINED= [[[[1, 1], [1, 1], [1, 1], [1, 2]], [[1, 5], [1, 5], [1, 10], [3, 4]], [[1, 1], [
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
