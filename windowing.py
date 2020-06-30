import pickle
import numpy as np
from skimage.util.shape import view_as_windows


def create_windows_from_dictmatrix(dict_matrix, rows, window_size, stride):
    """
    Description: takes dictionary that contains x, y, z, nanos keys and transforms
                 this into an np array. Then creates windows from this matrix and returns
                 the windows

    Args:
        dict_matrix -- dictionary  to create windows from
        rows -- int (number of rows in a trace)
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of samples between the start of each window)

    Return:
        windowed_data -- np array (np array of the windowed data)
    """
    x = dict_matrix['x']
    y = dict_matrix['y']
    z = dict_matrix['z']
    nanos = dict_matrix['nanos']
    np_data = np.array([x, y, z, nanos], 'd')
    windowed_data = view_as_windows(np_data, (rows, window_size), step=stride)
    return windowed_data


def find_initial_lift_times(lift_windows, rows):
    """
    Description: returns the start and end times from lift_windows np array

    Args:
        lift_windows -- np array
        rows -- int (number of rows in a trace)

    Return:
        start -- int (start time in nanos)
        end -- int (end time in nanos)
    """
    start = lift_windows[0][rows-1][0]
    end = lift_windows[-1][-1][-1]
    return start, end


def find_lift_windows(windowed_data, divisor, threshold, return_lift_windows):
    """
    Description: takes the windowed data and calculates variance for each window.
                 Then calculates variance threshold and returns an np array of
                 the windows that meet the variance threshold.
    Args:
        windowed_data -- np array [[[[x1],[y1],[z1],[nanos1]], [[x2], [y2], [z2], [nanos2]]...]
        divisor -- int (will use divisor to calculate variance threshold so
                   that the threshold is a function of the max variance. If using a set threshold, make
                   divisor = 0. Threshold = (max variance)/divisor)
        threshold -- int (used at the variance threshold. If using a divisor to find the variance threshold,
                                 leave threshold = 0)
        return_lift_windows -- boolean: (True:  will return a list of lift_windows that meet the variance threshold)
                                        (False: will return just the start time of the lift_windows)

    Return:
        lift_windows -- np array (a np_array of windows that meet the desired threshold)
        -- or --
        start_time -- int (start time in nanos of the lift_windows)

     """
    if (threshold != 0):
        divisor = 0
    indices = find_indices_that_meet_threshold(
        windowed_data, threshold, divisor)

    windows = windowed_data[0]
    if (return_lift_windows):
        return get_lift_windows_from_indices(indices, windowed_data)
    else:
        if (indices.size == 0):
            return None
        index_of_start = indices[0]
        start_time = windows[index_of_start][0][3][0]
        return start_time


def get_lift_windows_from_indices(indices, windowed_data):
    """
	Description: Finds the first set of continuous windows from the indices array

	Args:
	    windowed_data -- np array
	    indices -- int (threshold for variance, if using a divisor set to 0 instead)

	Return:
	    lift_windows -- np_array (array of indices that correspond to windows in
	                   windowed_data that meet the variance threshold level)
	"""
    windows = windowed_data[0]
    if (indices.size <= 1):
        indices = indices[0]
    else:
        indices = np.squeeze(indices)
        indices_diff = np.diff(indices)
        end_time = np.argwhere(indices_diff != 1)
        if (end_time.size != 0):
            indices = indices[:end_time[0][0]+1]
    lift_windows = np.take(windows, indices, axis=0)
    return lift_windows


def find_indices_that_meet_threshold(windowed_data, threshold, divisor):
    """
    Description: Finds indices that meet the desired threshold level

    Args:
        windowed_data -- np array
        threshold -- int (threshold for variance, if using a divisor set to 0 instead)
        divisor -- int (divisor to set threshold, if using threshold set to 0 instead)


    Return:
        indices -- np_array (array of indices that correspond to windows in
                   windowed_data that meet the variance threshold level)
    """
    windows = windowed_data[0]
    windows_no_nanos = windows[:, :3]
    variances = np.var(windows_no_nanos, axis=2, ddof=1)
    var_sum = np.sum(variances, axis=1)
    if divisor == 0:
        variance_threshold = threshold
    else:
        variance_threshold = max(var_sum)/divisor
    indices = np.argwhere(var_sum > variance_threshold)
    return indices


def find_precise_start_time(lift_windows, rows, window_size, stride, variance_threshold):
    """
    Description: Finds a more precise start time by creating much smaller windows
                 within the first window found to contain part of the lift ouputted
                 from initial_find_lift.
    Args:
        lift_windows -- np array
        rows -- int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of smaples in between the start of each window)
        variane_threshold -- int (variance threshold that determines
                             wether the current window is part of the lift)

    Return:
        start_time = int (more precise start time of the lift in nanoseconds)
    """
    front = lift_windows[0]
    divisor = 0
    return_lift_windows = False
    windowed_data = view_as_windows(front, (rows, window_size), stride)
    start_time = find_lift_windows(
        windowed_data, divisor, variance_threshold, return_lift_windows)
    return start_time


def initial_find_lift(sample, rows, window_size, stride, divisor):
    """
    Description: Finds set of windows whose varaince meets a threshold indicating it contains part of the 'lift'

    Args:
        sample -- dictionary matrix with the x, y, z, and nanos data points
        rows -- int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of smaples in between the start of each window)
        divisor -- int (will be used to find variacne threshold : threshold = (max variance)/divisor)

    Return:
        lift_windows -- np array (an array of windows that meet the variance threshold)
    """
    windowed_data = create_windows_from_dictmatrix(sample, rows, window_size, stride)
    return_lift_windows = True
    threshold = 0
    lift_windows = find_lift_windows(
        windowed_data, divisor, threshold, return_lift_windows)
    return lift_windows


def cropped_np(np_sample, start, end):
    """
	Description: returns the cropped sample from the start and end time in nanos

	Args:
	    np_sample -- np array
	    start -- int (start time in nanos to begin crop)
	    end -- int (end time in nanos the end crop)

	Return:
	    cropped -- np array (a np array from start to end time in nanos)
	"""
    nanos = np_sample[3]
    start_index = np.where(nanos == start)
    end_index = np.where(nanos == end)
    cropped = [np_sample[:,  start_index[0][0]:end_index[0][0] + 1]]
    return cropped
