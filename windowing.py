import pickle
import numpy as np
from skimage.util.shape import view_as_windows
import statistics

# Input: dict_matrix = dict
# Input: rows = int (number of rows in a trace)
# Input: window_size = int (# of samples within one window when creating the windows)
# Input: stride = int (# of samples between the start of each window)
#
# Description: takes dictionary that contains x, y, z, nanos keys and transforms
#			  this into an np array. Then creates windows from this matrix and returns
#			  the windows
#
# Return: windowed_data = np array (np array of the windowed data)


def create_windows_from_dictmatrix(dict_matrix, rows, window_size, stride):
    x = dict_matrix['x']
    y = dict_matrix['y']
    z = dict_matrix['z']
    nanos = dict_matrix['nanos']
    np_data = np.array([x, y, z, nanos], 'd')
    windowed_data = view_as_windows(np_data, (rows, window_size), step=stride)
    return windowed_data

# Input: lift_windows = np array
# Input: rows = int (number of rows in a trace)
#
# Description: returns the start and end times from lift_windows np array
#
# Returns: int start (start time in nanos)
#		   int end (end time in nanos)


def find_initial_lift_times(lift_windows, rows):
    start = lift_windows[0][rows-1][0]
    end = lift_windows[-1][-1][-1]
    return start, end

# Input: windowed_data = np array
# Input: divisor = int (will use divisor to calculate variance threshold so
#						that the threshold is a function of the max variance.
#						Threshold = (max variance)/divisor)
#
# Description: takes the windowed data and calculates variance for each window.
#				Then calculates variance threshold and returns an np array of
#				the windows that meet the variance threshold.
#
# Return: lift_windows = np array


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
    lift_windows = np.take(windows, indices, axis=0)
    return lift_windows

# Input: windowed_data = np array
# Input: threshold = int (variance threshold for start of lift
#
# Description: finds the more refined start time by calcualting variacne
#				on the smaller windows and returning the start time of the first
#				window that meets the thresholdusing a smaller threshold.
#
# Return: refined start time = int (in nanos)


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

# Input: lift_windows = np array
# Input: rows = int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
# Input: window_size = int (# of samples within one window when creating the windows)
# Input: stride = int (# of smaples in between the start of each window)
# Input: variane_threshold = int (variance threshold that determines
#								  wether the current window is part of the lift)
#
# Description: Finds the more precise start time by creating much smaller windows
#			  within the first window found to contain part of the lift ouputted
#			  from initial_find_lift.
#
# Return: start_time = int (more precise start time of the lift in nanoseconds)


def find_precise_start_time(lift_windows, rows, window_size, stride, variance_threshold):
    front = lift_windows[0]
    np_front = np.array(front)
    windowed_data = view_as_windows(np_front, (rows, window_size), step=stride)
    start_time = refine_start_time(windowed_data, variance_threshold)
    return start_time

# Input: lift_windows = np array
# Input: rows = int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
# Input: window_size = int (# of samples within one window when creating the windows)
# Input: stride = int (# of smaples in between the start of each window)
# Input: divisor = int (will be used to find variacne threshold : threshold = (max variance)/divisor)
#
# Description: Finds set of windows whose varaince meets a threshold indicating it contains part of the 'lift'
#
# Return: lift_windows = np array (an array of windows that meet the variance threshold)


def initial_find_lift(sample, rows, window_size, stride, divisor):
    windowed_data = create_windows(sample, rows, window_size, stride)
    lift_windows = find_lift_windows(windowed_data, divisor)
    return lift_windows
