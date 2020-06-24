import pickle
import numpy as np
from skimage.util.shape import view_as_windows
import statistics


def create_windows(dict_matrix, rows, window_size, stride):
    x = dict_matrix['x']
    y = dict_matrix['y']
    z = dict_matrix['z']
    nanos = dict_matrix['nanos']
    np_data = np.array([x, y, z, nanos], 'd')
    windowed_data = view_as_windows(np_data, (rows, window_size), step=stride)
    return windowed_data


def find_initial_lift_times(lift_windows, rows):
    start = lift_windows[0][rows-1][0]
    end = lift_windows[-1][-1][-1]
    return start, end


def find_lift_windows(windowed_data, divisor):
    windows = windowed_data[0]
    windows_no_nanos = np.delete(windows, 3, 1)
    variances = np.var(windows_no_nanos, axis=2, ddof=1)
    var_sum = np.sum(variances, axis=1)
    indices = np.argwhere(var_sum > max(var_sum)/6)
    if (len(indices) <= 1):
        indices = indices[0]
    else:
        indices = np.squeeze(indices)
        count = indices[0]-1
        for i in range(len(indices)):
            count += 1
            if count != indices[i]:
                end = i
                break
        indices = indices[:i]
    lift = np.take(windows, indices, axis=0)
    return lift


def refine_start_time(windowed_data, threshold):
    windows = windowed_data[0]
    windows_no_nanos = np.delete(windowed_data[0], 3, 1)
    variances = np.var(windows_no_nanos, axis=2, ddof=1)
    var_sum = np.sum(variances, axis=1)
    indexes = np.argwhere(var_sum > threshold)
    if (len(indexes) < 1):
        return None
    index_of_start = indexes[0]
    return windows[index_of_start][0][3][0]


def find_precise_start_time(lift_windows, rows, window_size, stride, variance_threshold):
    front = lift_windows[0]
    np_front = np.array(front)
    windowed_data = view_as_windows(np_front, (rows, window_size), step=stride)
    start_time = refine_start_time(windowed_data, variance_threshold)
    return start_time


def initial_find_lift(sample, rows, window_size, stride, divisor):
    windowed_data = create_windows(sample, rows, window_size, stride)
    lift_windows = find_lift_windows(windowed_data, divisor)
    return lift_windows
