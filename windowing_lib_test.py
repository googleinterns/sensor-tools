import windowing_library as wl
import numpy as np
import pytest

windowed_data = [[[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, ], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]], [
    [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]]
np_wd = np.array(windowed_data, 'd')
sample_matrix = {'x': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'y': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'z': [
    1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'nanos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


def test_create_windows():
    window_size = 5
    stride = 5

    test_windowed_data = wl.create_windows(
        sample_matrix, 4, window_size, stride)
    assert (test_windowed_data == np_wd).all()


lift_windows = [[[20, 20, 1, 1, 1], [20, 20, 1, 1, 1],
                 [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]


def test_find_intial_lift_times():
    exp_start = 6
    exp_end = 10
    start, end = wl.find_initial_lift_times(lift_windows, 4)
    assert start == exp_start
    assert end == exp_end


def test_find_lift_windows():
    divisor = 4
    actual_lift_windows = np.array(wl.find_lift_windows(np_wd, divisor), 'd')
    np_lift_windows = np.array(lift_windows, 'd')
    assert (np_lift_windows == actual_lift_windows).all()


windowed_data_refined = [[[[1, 1], [1, 1], [1, 1], [1, 2]], [[1, 5], [1, 5], [1, 10], [3, 4]], [[1, 1], [
    1, 1], [1, 1], [5, 6]], [[1, 20], [1, 20], [1, 20], [7, 8]], [[20, 20], [20, 20], [20, 20], [9, 10]]]]


def test_refine_start_time():
    np_windowed_data_refined = np.array(windowed_data_refined)
    threshold = 2
    exp_start_time = 3
    actual_start_time = wl.refine_start_time(np_windowed_data_refined, threshold)
    assert actual_start_time == exp_start_time


front_window_lift = [[[1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 1, 2, 2, 2, 2, 20, 20, 20], [
    1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9]]]


def test_find_precise_start_time():
    np_front_window_lift = np.array(front_window_lift, 'd')
    rows = 4
    window_size = 2
    stride = 1
    threshold = 5
    exp_start_time = 6
    actual_start_time = wl.find_precise_start_time(
        np_front_window_lift, rows, window_size, stride, threshold)
    assert exp_start_time == actual_start_time
