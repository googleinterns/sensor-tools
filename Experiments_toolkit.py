import windowing as wl
import numpy as np
from skimage.util.shape import view_as_windows
import tensorflow.data as tfdata


def create_train_test_validation(positives, negatives, percent_train, percent_test, percent_validation):
    """
        Description: creates a training, testing, and validation dictionary for binary data

        Args:
            positives -- np_array (List of positive samples as numpy arrays)
            negatives -- np_array (List of negative samples as numpy arrays)
            percent_train -- double ([0-1] Fraction of data to go to training set)
            percent_test -- double (0-1] Fraction of data to go to testing set)
            percent_validation -- double ([0-1] Fraction of data to go to validation set)

        Return:
            train -- dict (Dictionary with percent_train of original dictionary)
            test -- dict (Dictionary with percent_test of original dictionary)
            validation -- dict (Dictionary with percent_validation of original dictionary)
        """
    train = {"postivies": [], "negatives": []}
    test = {"postivies": [], "negatives": []}
    validation = {"postivies": [], "negatives": []}

    if (percent_train + percent_test + percent_validation != 1):
        print("ERROR: percentages don't add to 1")

    train, test, validation = splitting_up_data(
        positives, "positives", train, percent_train, test, percent_test, validation, percent_validation)
    train, test, validation = splitting_up_data(
        negatives, "negatives", train, percent_train, test, percent_test, validation, percent_validation)

    return train, test, validation


def splitting_up_data(data, key, train, percent_train, test, percent_test, validation, percent_validation):
    """
        Description: Splits up data from dictionary using the given percentages to make a train, test, and validation set

        Args:
            data -- np array (Array of samples as numpy arrays)
            key -- String (Key for the in dictionary from which we will split up the data from)
            train -- dict (Training dictionary )
            percent_train -- double ([0-1] Percentage of data to go to training set)
            test -- dict (Testing dictionary )
            percent_test -- double (0-1] Percentage of data to go to testing set)
            validation -- dict (Validation dictionary )
            percent_validation -- double ([0-1] Percentage of data to go to validation set)

        Return:
            train -- dict (Dictionary with percent_train of list)
            test -- dict (Dictionary with percent_test of list)
            validation -- dict (Dictionary with percent_validation of list)
        """
    length = len(data)
    train_index = int(length * percent_train)
    test_index = int(length * percent_test)
    train[key] = data[0:train_index]
    test[key] = data[train_index:train_index + test_index]
    validation[key] = data[train_index + test_index:]
    return train, test, validation


def preprocess_data(data, rows, filtering_params, window_function, window_size, offset):
    """
       Description: Preprocess the data stored in data by windowing the traces and saving
                                these windows (numpy arrays) into a dictionary.

       Args:
            data -- list (List of numpy arrays where the last row is time steps, 
            			  [(sensor_dims + 1), num_samples])
            rows -- int (Number of rows in the data)
            filtering_params -- All_Filtering_Params (Dataclass found in windowing.py that 
                                stores important parameters for the 2 steps involved with 
                                finding the precise start and end time of a activity in the data)
            window_size -- int (Window size of the trace to be saved into the
            				    dictionary that will be returned)
            offset -- int (Number to add to the predicted start time of the 
            			   activity from where the window will start at)

       Return:
           windows-- int[] (Array of the windows cropped by the window_function)
           """
    windows = []
    for j in range(len(data)):
        sample = data[j]
        activity_windows = wl.initial_find_activity(
            sample, rows, filtering_params.initial_filtering.window_size,
            filtering_params.initial_filtering.stride,
            filtering_params.initial_filtering.divisor)
        if len(activity_windows) > 0:
            start_time, end_time = wl.get_start_time(
                activity_windows, rows, filtering_params.precise_filtering.window_size,
                filtering_params.precise_filtering.stride,
                filtering_params.precise_filtering.threshold)
            pos_window = window_function(
                sample, start_time + offset, window_size)
            if len(pos_window[0][0]) != window_size:
                continue
            windows.extend(pos_window)
    np_windows = np.array(windows)
    return np_windows


def get_np_X_Y(x, y, length):
    """
       Description: Create X and Y numpy arrays desired length of random 
       				samples from lists of windowed data, x and y

       Args:
           x -- np_array (Numpy array of one category numpy arrays with the same shape)
           y -- np_array (Numpy array of another category numpy arrays with 
           				  the same shape as x numpy arrays)
           length -- int Number of samples in the numpy array to be returned 

       Return:
            X -- np array (Numpy array where the first half of the data points
            			  come from x and second half comes from y)
            y -- np array (Numpy array with length = length with binary labels 
            			   (1, 0) for the differetn x, y data points)
           """
    len1 = len(x)
    len2 = len(y)
    limit = min(len1, len2, length)
    idx = np.random.randint(len1, size=limit)
    idx2 = np.random.randint(len2, size=limit)
    rnd_x = x[idx]
    rnd_y = y[idx2]
    X = np.concatenate((rnd_x, rnd_y))
    pos = np.ones(limit)
    neg = np.zeros(limit)
    y = np.concatenate((pos, neg))
    return X, y


def create_dataset(pos, neg, max_samples):
    """
       Description: Create a tf.data.Dataset composed of samples from the pos and neg list 

       Args:
           pos -- np array (Numpy arrays of positive windows, [(sensor_dims), num_samples])
           neg -- np arrya (Numpy arrays of negative windows, [(sensor_dims), num_samples])
           max_size -- int (number of samples to be taken from pos and neg to
                                                be used in the dataset. If max_size < 0, then all 
                                                samples from both pos and neg will be used) 

       Return:
           dataset -- tf.data.Dataset (TensorFlow dataset composed of the 
                                  positive and negative lists of data passed into the 
                                  function with the corresponding labels)
     """
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    if (max_samples > 0):
        pos = pos[:max_samples]
        neg = neg[:max_samples]
    examples = np.concatenate((pos, neg))
    pos_label = np.ones(len(pos))
    neg_label = np.zeros(len(neg))
    labels = np.concatenate((pos_label, neg_label))
    dataset = tfdata.Dataset.from_tensor_slices((examples, labels))
    return dataset


def preprocess_add_augmentation(data, augmentation_function, sigma):
    """
       Description: Iterate over DATA and proprocess the traces by adding an augmentation function
                                to the data.

       Args:
           data -- list (List of numpy array where the last row is time steps, [(sensor_dims + 1), num_samples])
           dictionary -- dict (Dictionary to store augmented traces)
           pos_key -- string (Key in dictionary to store augmented windows) 
           augmentation_function -- function (Desired augmentation function of the form:
                                              augment(np_array data, int sigma))
           sigma -- double (Level of augment to be passed to the augmentation function )
           size -- int (Window size of traces to be stored in dictionary)

       Return:
           dictionary -- dict (Dictionary with augmented windows of size = size in key pos_key)
     """
    augmented = []
    for j in range(len(data)):
        sample = data[j]
        sample = sample[:-1]
        transposed = sample.transpose()
        augmented_trace = augmentation_function(transposed, sigma)
        augmented_trace = augmented_trace.transpose()
        augmented.extend([augmented_trace])
    return augmented
