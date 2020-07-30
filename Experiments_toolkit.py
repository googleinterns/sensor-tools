import windowing as wl
import numpy as np
from skimage.util.shape import view_as_windows
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


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
            data -- list (List of numpy arrays where the last row is time steps, [(sensor_dims + 1), num_samples])
            rows -- int (Number of rows in the data)
            filtering_params -- All_Filtering_Params (Dataclass found in windowing.py that 
                                                        stores important parameters for the 2 steps involved with 
                                                        finding the precise start and end time of a lift in the data)
            window_size -- int (Window size of the trace to be saved into the dictionary that will be
                                                returned)
            offset -- int (Number to add to the predicted start time of the lift from where the window
                                                will start at)

       Return:
           windows-- int[] (Array of the windows cropped by the window_function)
           """
    windows = []
    for j in range(len(data)):
        sample = data[j]
        lift_windows = wl.initial_find_lift(
            sample, rows, filtering_params.initial_filtering.window_size, 
            filtering_params.initial_filtering.stride, 
            filtering_params.initial_filtering.divisor)
        if len(lift_windows) > 0:
            start_time, end_time = wl.get_start_time(
                lift_windows, rows, filtering_params.precise_filtering.window_size, 
                filtering_params.precise_filtering.stride, 
                filtering_params.precise_filtering.threshold)
            pos_window = window_function(
                sample, start_time + offset, window_size)
            if len(pos_window[0][0]) != window_size:
                continue
            windows.extend(pos_window)
    windows = np.array(windows)
    return windows


def get_np_X_Y(x, y, length):
    """
       Description: Create X and Y numpy arrays desired length of random samples from lists of windowed data, x and y

       Args:
           x -- np_array (Numpy array of one category numpy arrays with the same shape)
           y -- np_array (Numpy array of another category numpy arrays with the same shape as x numpy arrays)
           length -- int Number of samples in the numpy array to be returned 

       Return:
            X -- np array (Numpy array where the first half of the data points come from x and second
                                          half comes from y)
            y -- np array (Numpy array with length = length with binary labels (1, 0) for the differetn x, y 
                                          data points)
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


def preprocess_with_augmentation(data, dictionary, pos_key, augmentation_function, sigma, size):
    """
       Description: Iterate over DATA and proprocess the traces by adding an augmentation function
                                to the data.

       Args:
           data -- np array (Numpy array where the last row is time steps, [(sensor_dims + 1), num_samples])
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
        length = len(np_sample[0][0])
        if length < size:
            continue
        window = same_window_size(sample[:-1], size, 10)
        transpose = window.transpose(0, 2, 1)
        for i in range(len(transpose)):
            augmented_trace = augmentation_function(transpose[i], sigma)
            augmented.extend(augmented_trace)
    return augmented


def DA_Rotation_specific(X, angle_low, angle_high, axis):
    axis = np.array(axis)
    angle = np.random.uniform(low=angle_low, high=angle_high)
    return np.matmul(X, axangle2mat(axis, angle))


def scatter_PCA(X, Y, alpha):
    components = len(X[0])
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(X)
    scatter_plot(pca_result, Y, alpha)


def scatter_ICA(X, Y, alpha):
    components = len(X[0])
    ica = FastICA(n_components=components)
    ica_result = ica.fit_transform(X)
    scatter_plot(ica_result, Y, alpha)


def scatter_TSNE(X, Y, alpha):
    components = len(X[0])
    RS = 20150101
    TSNE_proj = TSNE(random_state=RS, n_components=components).fit_transform(X)
    scatter_plot(TSNE_proj, Y, alpha)


def scatter_plot(result, Y, alpha):
    df_subset = {"negative": [], "positive": []}
    df_subset['positive'] = result[:, 0]
    df_subset['negative'] = result[:, 1]
    df_subset['y'] = Y
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="negative", y="positive",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=alpha
    )
