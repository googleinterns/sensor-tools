import windowing as wl
import numpy as np
import pytest
import pytest_diff


SAMPLE_MATRIX = {'x': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'y': [1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'z': [
    1, 1, 1, 1, 1, 20, 20, 1, 1, 1], 'nanos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
WINDOWED_DATA = [[[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, ], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]], [
    [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]]
NP_WINDOWED_DATA = np.array(WINDOWED_DATA, 'd')

def test_dict_to_numpy_array():
    expected_np = np.array([[1,1,1,1,1,20,20,1,1,1],[1,1,1,1,1,20,20,1,1,1],[1,1,1,1,1,20,20,1,1,1],[1,2,3,4,5,6,7,8,9,10]])

    actual_np = wl.dict_to_numpy_array(SAMPLE_MATRIX)
    assert (expected_np==actual_np).all()

def test_create_windows():

    window_size = 5
    stride = 5

    test_windowed_data = wl.create_windows_from_dictmatrix(
        SAMPLE_MATRIX, 4, window_size, stride)
    assert (test_windowed_data == NP_WINDOWED_DATA).all()


activity_WINDOWS = [[[20, 20, 1, 1, 1], [20, 20, 1, 1, 1],
                 [20, 20, 1, 1, 1], [6, 7, 8, 9, 10]]]


def test_find_intial_activity_times():
    expected_start = 6
    expected_end = 10
    npactivity = np.array(activity_WINDOWS)
    start, end = wl.find_initial_activity_times(npactivity, 4)
    assert start == expected_start
    assert end == expected_end


def test_find_activity_windows():
    divisor = 4
    threshold = 0
    return_activity_windows = True
    l_windows = wl.find_activity_windows(
        NP_WINDOWED_DATA, divisor, threshold, return_activity_windows)
    actual_activity_windows = np.array(l_windows, 'd')
    np_activity_windows = np.array(activity_WINDOWS, 'd')
    assert (np_activity_windows == actual_activity_windows).all()


def test_refined_start_time():
    WINDOWED_DATA_REFINED = [[[[1, 1], [1, 1], [1, 1], [1, 2]], [[1, 5], [1, 5], [1, 10], [3, 4]], [[1, 1], [
        1, 1], [1, 1], [5, 6]], [[1, 20], [1, 20], [1, 20], [7, 8]], [[20, 20], [20, 20], [20, 20], [9, 10]]]]

    NP_WINDOWED_DATA_REFINED = np.array(WINDOWED_DATA_REFINED)
    threshold = 2
    divisor = 0
    return_activity_windows = False
    expected_start_time = 3
    actual_start_time = wl.find_activity_windows(
        NP_WINDOWED_DATA_REFINED, divisor, threshold, return_activity_windows)
    assert actual_start_time == expected_start_time


def test_find_precise_start_time():
    FRONT_WINDOW_activity = [[[1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 1, 2, 2, 2, 2, 20, 20, 20], [
        1, 1, 2, 2, 2, 2, 20, 20, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9]]]
    np_front_window_activity = np.array(FRONT_WINDOW_activity, 'd')
    rows = 4
    window_size = 2
    stride = 1
    threshold = 5
    expected_start_time = 6
    actual_start_time = wl.find_precise_start_time(
        np_front_window_activity, rows, window_size, stride, threshold)
    assert expected_start_time == actual_start_time


FRONT_WINDOW_activity = [[[1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [
    1, 1, 2, 2, 2, 2, 20, 20, 20, 20], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]]
NP_FRONT_WINDOW_activity = np.array(FRONT_WINDOW_activity)


def test_get_start_time():
    expected_start = 60
    expected_end = 100
    rows = 4
    window_size = 2
    stride = 1
    variance_threshold = 6
    actual_start, actual_end = wl.get_start_time(
        NP_FRONT_WINDOW_activity, rows, window_size, stride, variance_threshold)
    assert expected_start == actual_start
    assert expected_end == actual_end


def test_centered_window1():
    expected_window = np.array(
        [[2, 2, 2, 2, 20], [2, 2, 2, 2, 20], [2, 2, 2, 2, 20], [30,40,50,60,70]])
    center_timestamp = 50
    window_size = 5
    actual_window = wl.centered_window(
        NP_FRONT_WINDOW_activity[0], center_timestamp, window_size)
    assert (expected_window == actual_window).all()


def test_centered_window2():
    expected_window = np.array(
        [[2, 20, 20, 20, 20], [2, 20, 20, 20, 20], [2, 20, 20, 20, 20], [60,70,80,90,100]])
    center_timestamp = 90
    window_size = 5
    actual_window = wl.centered_window(
        NP_FRONT_WINDOW_activity[0], center_timestamp, window_size)
    assert (expected_window == actual_window).all()


def test_centered_window3():
    expected_window = np.array(
        [[1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2], [10,20,30,40,50,60]])
    center_timestamp = 20
    window_size = 6
    actual_window = wl.centered_window(
        NP_FRONT_WINDOW_activity[0], center_timestamp, window_size)
    assert (expected_window == actual_window).all()


def test_centered_window4():
    expected_window = np.array(
        [[1, 1, 2, 2, 2, 2, 20, 20, 20, 20],
         [1, 1, 2, 2, 2, 2, 20, 20, 20, 20],
         [1, 1, 2, 2, 2, 2, 20, 20, 20, 20], 
         [10,20,30,40,50,60,70,80,90,100]])
    center_timestamp = 50
    window_size = 20
    actual_window = wl.centered_window(
        NP_FRONT_WINDOW_activity[0], center_timestamp, window_size)
    assert (expected_window == actual_window).all()


def test_get_window_from_timestamp():
    expected_window = np.array(
        [[2, 20, 20],
         [2, 20, 20],
         [2, 20, 20],
         [60,70,80]])
    start_timestamp = 60
    window_size = 3
    actual_window = wl.get_window_from_timestamp(
        NP_FRONT_WINDOW_activity[0], start_timestamp, window_size)
    print(actual_window)
    assert (expected_window == actual_window).all()


def test_get_window_from_timestamp2():
    expected_window = np.array(
        [[2,  2,  2,  2, 20, 20, 20, 20],
         [2,  2,  2,  2, 20, 20, 20, 20],
         [2,  2,  2,  2, 20, 20, 20, 20],
         [30,40,50,60,70,80,90, 100]])
    start_timestamp = 60
    window_size = 8
    actual_window = wl.get_window_from_timestamp(
        NP_FRONT_WINDOW_activity[0], start_timestamp, window_size)
    assert (expected_window == actual_window).all()
