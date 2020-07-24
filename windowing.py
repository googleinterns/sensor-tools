import numpy as np
from skimage.util.shape import view_as_windows
from dataclasses import dataclass

@dataclass
class Initial_Filtering_Params:
    window_size: int
    stride: int
    divisor: int
    rows: int 

@dataclass
class Precise_Filtering_Params:
    window_size: int
    stride: int
    threshold: int
    rows: int

@dataclass
class All_Filtering_Params:
    intial_filtering: Initial_Filtering_Params
    precise_filtering: Precise_Filtering_Params


def dict_to_numpy_array(dictionary):
    """
    Description: converts dictionary of raw data to and Numpy array. 

    Args:
        dictionary -- Dictionary of data to be converted to Numpy array. All the entries
                      within the keys must be the same length for the Numpy.append function
                      to work.

    Return:
        np_array -- np array (Np array the data in dictionary)
    """
    np_array = np.array([[]])
    i = 0
    for key in dictionary:
        if (i == 0):
            np_array = np.array([dictionary[key]])
        else:
            np_array = np.append(np_array, [dictionary[key]], axis=0)
        i += 1
    return np_array


def create_windows_from_dictmatrix(dict_matrix, rows, window_size, stride):
    """
    Description: Takes dictionary that contains x, y, z, nanos keys and transforms
                 this into an np array. Then creates windows from this matrix and returns
                 the windows

    Args:
        dict_matrix -- Dictionary to create windows from
        rows -- int (Number of rows in a trace)
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of samples between the start of each window)

    Return:
        windowed_data -- np array (Np array of the windowed data)
    """

    np_data = dict_to_numpy_array(dict_matrix)
    windowed_data = view_as_windows(np_data, (rows, window_size), step=stride)
    return windowed_data


def find_initial_lift_times(lift_windows, rows):
    """
    Description: Returns the start and end times from lift_windows np array

    Args:
        lift_windows -- np array
        rows -- int (Number of rows in a trace)

    Return:
        start -- int (Start time in nanos)
        end -- int (End time in nanos)
    """
    start = lift_windows[0][rows-1][0]
    end = lift_windows[-1][-1][-1]
    return start, end


def find_lift_windows(windowed_data, divisor, threshold, return_lift_windows):
    """
    Description: Takes the windowed data and calculates variance for each window.
                 Then calculates variance threshold and returns an np array of
                 the windows that meet the variance threshold.
    Args:
        windowed_data -- np array [[[[x1],[y1],[z1],[nanos1]], [[x2], [y2],
                                         [z2], [nanos2]]...]
        divisor -- int (Will use divisor to calculate variance threshold so
                   that the threshold is a function of the max variance. If using
                   a set threshold, make divisor = 0.
                   Threshold = (max variance)/divisor)
        threshold -- int (Used as the variance threshold. If using a divisor to
                          find the variance threshold, leave threshold = 0)
        return_lift_windows -- boolean: (True:  will return a list of
                                         lift_windows that meet the variance threshold)
                                        (False: will return just the start time of the lift_windows)

    Return:
        lift_windows -- np array (A np_array of windows that meet the desired
                                  threshold)
        -- or --
        start_time -- int (Start time in nanos of the lift_windows)

     """
    if (threshold != 0 and divisor != 0):
        print("ERROR: Either divisor or threshold need equal to 0 to specify which to use.")
        return -1
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
        Description: Finds the first set of continuous windows from the indices
                                 array

        Args:
            windowed_data -- np array
            indices -- int (Threshold for variance, if using a divisor set to 0
                                        instead)

        Return:
            lift_windows -- np_array (Array of indices that correspond to windows in
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
        threshold -- int (Threshold for variance, if using a divisor set to 0 instead)
        divisor -- int (Divisor to set threshold, if using threshold set to 0 instead)


    Return:
        indices -- np_array (Array of indices that correspond to windows in
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


def find_precise_start_time(lift_windows, rows,
                            window_size, stride, variance_threshold):
    """
    Description: Finds a more precise start time by creating much smaller windows
                 within the first window found to contain part of the lift ouputted
                 from initial_find_lift.
    Args:
        lift_windows -- np array
        rows -- int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of smaples in between the start of each window)
        variane_threshold -- int (Variance threshold that determines
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
    Description: Finds set of windows whose variance meets a
                         threshold indicating it contains part of the 'lift'

    Args:
        sample -- Dictionary matrix with the x, y, z, and nanos data points
        rows -- int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
        window_size -- int (# of samples within one window when creating the windows)
        stride -- int (# of smaples in between the start of each window)
        divisor -- int (Will be used to find variacne threshold : threshold = (max variance)/divisor)

    Return:
        lift_windows -- np array (an array of windows that meet the variance threshold)
    """
    windowed_data = create_windows_from_dictmatrix(
        sample, rows, window_size, stride)
    return_lift_windows = True
    threshold = 0
    lift_windows = find_lift_windows(
        windowed_data, divisor, threshold, return_lift_windows)
    return lift_windows


def cropped_np(np_sample, start, end):
    """
        Description: Returns the cropped sample from the start and end time in nanos

        Args:
            np_sample -- np array (Np array to be cropped)
            start -- int (Start time in nanos to begin crop)
            end -- int (End time in nanos the end crop)

        Return:
            cropped -- np array (a np array from start to end time in nanos)
        """
    nanos = np_sample[3]
    start_index = np.where(nanos == start)
    end_index = np.where(nanos == end)
    cropped = [np_sample[:,  start_index[0][0]:end_index[0][0] + 1]]
    return cropped


def get_start_time(lift_windows, rows, window_size, stride, variance_threshold):
    """
        Description: Returns the precise start time

        Args:
            lift_windows -- np_array (Np array of the lift windows)
            rows -- int (# of rows within a trace (ex: x, y, z, nanos => 4 rows))
            window_size -- int (# of samples within one window when creating the windows)
            stride -- int (# of samples to jump before creating next window)
            variance_threshold -- int (Variance threshold that determines
                                  wether the current window is part of the lift)

        Return:
            start_time -- int (start time in nanos)
            end -- int (end time in nanos)
        """
    start, end = find_initial_lift_times(lift_windows, rows)
    precise_start = find_precise_start_time(
        lift_windows, rows, window_size, stride, variance_threshold)
    if (precise_start != None):
        start_time = precise_start
    else:
        start_time = start
    return start_time, end


def centered_window(np_sample, center_time, window_size):
    """
    Description: Creates a window centered around the center_time of the lift

    Args:
        np_sample -- np array (The full trace as an np array)
        center_time -- int (Time in nanoseconds that the window should be centered around)
        window_size -- int (Size of window)


    Return:
        window -- np_array (Window centered around the given time)
    """
    nanos = np_sample[3]
    center_index = np.argwhere(nanos == center_time)[0][0]
    add_front = int(window_size/2)
    add_back = window_size - add_front
    back_index = center_index + add_back
    front_index = center_index - add_front
    last = len(np_sample[0])-1
    if (front_index < 0):
        add_back += abs(front_index)
        front_index = 0
        back_index = center_index + add_back
    if (back_index > last):
        add_front = abs(last - back_index) + add_front - 1
        back_index = last + 1
        front_index = center_index - add_front
        if (front_index < 0):
            front_index = 0
    window = [np_sample[:3, front_index: back_index]]
    return window


def get_window_from_timestamp(np_sample, start_time, window_size):
    """
    Description: Creates a window that begins at start_time and has a length = window_size.
                 Additionally, if the start_time is negative or the window size means
                 that the window will go out of range of np_sample, this function will 
                 wrap around and create a window of the same size that is just bounded by
                 np_sample length

    Args:
        np_sample -- np array (The full trace as an np array)
        start_time -- int (Time in nanoseconds that the window should start at)
        window_size -- int (Size of window)


    Return:
        window -- np_array (Window centered around the given time)
    """
    nanos = np_sample[3]
    start_index = np.argwhere(nanos == start_time)[0][0]
    add_back = window_size
    back_index = start_index + add_back
    front_index = start_index
    last = len(np_sample[0])-1
    if (front_index < 0):
        add_back += abs(front_index)
        front_index = 0
        back_index = stat_index + add_back
    if (back_index > last):
        add_front = abs(last - back_index) - 1
        back_index = last + 1
        front_index = start_index - add_front
        if (front_index < 0):
            front_index = 0
    window = [np_sample[:3, front_index: back_index]]
    return window
