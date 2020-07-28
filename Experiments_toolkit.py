import windowing as w
import numpy as np
from skimage.util.shape import view_as_windows
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def create_train_test_validation(dictionary, pos_key, neg_key, percent_train, percent_test, percent_validation):
    """
        Description: creates a training, testing, and validation dictionary for binary data

        Args:
            dictionary -- dict (Dictionary of data sorted into pos_key and neg_key:
                                dictionary = {pos_key: [data, data,...], neg_key: [data, data, ...]})
            pos_key -- String (Key for the positive data samples in dictionary)
            neg_key -- String (Key for the negative data samples in dictionary)
            percent_train -- double ([0-1] Fraction of data to go to training set)
            percent_test -- double (0-1] Fraction of data to go to testing set)
            percent_validation -- double ([0-1] Fraction of data to go to validation set)

        Return:
            train -- dict (Dictionary with percent_train of original dictionary)
            test -- dict (Dictionary with percent_test of original dictionary)
            validation -- dict (Dictionary with percent_validation of original dictionary)
        """
    train = {pos_key: [], neg_key: []}
    test = {pos_key: [], neg_key: []}
    validation = {pos_key: [], neg_key: []}

    if (percent_train + percent_test + percent_validation != 1):
        print("ERROR: percentages don't add to 1")

    train, test, validation = splitting_up_data(
        dictionary, pos_key, train, percent_train, test, percent_test, validation, percent_validation)
    train, test, validation = splitting_up_data(
        dictionary, neg_key, train, percent_train, test, percent_test, validation, percent_validation)

    return train, test, validation


def splitting_up_data(dictionary, dict_key, train, percent_train, test, percent_test, validation, percent_validation):
    """
        Description: Splits up data from dictionary using the given percentages to make a train, test, and validation set

        Args:
            dictionary -- dict (Dictionary of data sorted into pos_key and neg_key:
                                                dictionary = {pos_key: [data, data,...], neg_key: [data, data, ...]})
            dict_key -- String (Key for the in dictionary from which we will split up the data from)
            train -- dict (Training dictionary )
            percent_train -- double ([0-1] Percentage of data to go to training set)
            test -- dict (Testing dictionary )
            percent_test -- double (0-1] Percentage of data to go to testing set)
            validation -- dict (Validation dictionary )
            percent_validation -- double ([0-1] Percentage of data to go to validation set)

        Return:
            train -- dict (Dictionary with percent_train of dictionary[dict_key])
            test -- dict (Dictionary with percent_test of dictionary[dict_key])
            validation -- dict (Dictionary with percent_validation of dictionary[dict_key])
        """
    length = len(dictionary[dict_key])
    train_index = int(length * percent_train)
    test_index = int(length * percent_test)
    train[dict_key] = dictionary[dict_key][0:train_index]
    test[dict_key] = dictionary[dict_key][train_index:train_index + test_index]
    validation[dict_key] = dictionary[dict_key][train_index + test_index:]
    return train, test, validation


def preprocess_data(DATA, rows, filtering_params, window_size, offset):
    """
       Description: Preprocess the data stored in DATA by windowing the traces and saving
                                these windows (numpy arrays) into a dictionary.

       Args:
           DATA -- dict (Dictionary of keys and within the keys are the 3D data samples)
           rows -- int (Number of rows in the data)
           filtering_params -- All_Filtering_Params (Dataclass found in windowing.py that 
                                                        stores important parameters for the 2 steps involved with 
                                                        finding the precise start and end time of a lift in the data)
            window_size -- int (Window size of the trace to be saved into the dictionary that will be
                                                returned)
                offset -- int (Number to add to the predicted start time of the lift from where the window
                                                will start at)

       Return:
           windowed_data_dictionary -- dict (Dictionary that stores the windows under the "windows" key)
           """
    windowed_data_dictionary = {"windows": []}
    for key in DATA:
        for j in range(len(DATA[key])):
            sample = DATA[key][j]
            np_sample = dict_to_np(sample)
            lift_windows = wl.initial_find_lift(
                sample, rows, filtering_params.initial_filtering.window_size, filtering_params.initial_filtering.stride, filtering_params.initial_filtering.divisor)
            if len(lift_windows) > 0:
                start_time, end = get_start_time(
                    lift_windows, rows, filtering_params.precise_filtering.window_size, filtering_params.precise_filtering.stride, filtering_params.precise_filtering.threshold)
                cropped = wl.cropped_np(np_sample, start_time, end)
                pos_window = wl.get_window_from_timestamp(
                    np_sample, start_time + offset, window_size)
                if len(pos_window[0][0]) != SIZE:
                    continue
                windowed_data_dictionary["windows"].extend(pos_window)
    return windowed_data_dictionary


def get_np_X_Y(x, y, length):
    """
       Description: Create X and Y numpy arrays desired length of random samples from lists of windowed data, x and y

       Args:
           x -- list (List of one category numpy arrays with the same shape)
           y -- list (List of another category numpy arrays with the same shape as x numpy arrays)
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
    np_x = np.array(x)
    np_y = np.array(y)
    rnd_x = np_x[idx]
    rnd_y = np_y[idx2]
    X = np.concatenate((rnd_x, rnd_y))
    pos = np.ones(limit)
    neg = np.zeros(limit)
    y = np.concatenate((pos, neg))
    return X, y


def preprocess_with_augmentation(DATA, dictionary, pos_key, augmentation_function, sigma, size):
    """
       Description: Iterate over DATA and proprocess the traces by adding an augmentation function
                                to the data.

       Args:
           DATA -- dict (Dictionary of keys and within the keys are the 3D data samples)
           dictionary -- dict (Dictionary to store augmented traces)
           pos_key -- string (Key in dictionary to store augmented windows) 
           augmentation_function -- function (Desired augmentation function of the form:
                                                                                        augment(np_array data, int sigma))
           sigma -- double (Level of augment to be passed to the augmentation function )
           size -- int (Window size of traces to be stored in dictionary)

       Return:
           dictionary -- dict (Dictionary with augmented windows of size = size in key pos_key)
     """
    for key in DATA:
        for j in range(len(DATA[key])):
            sample = DATA[key][j]
            np_sample = dict_to_np(sample)
            length = len(np_sample[0][0])
            if length < size:
                continue
            window = same_window_size(np_sample[:3], size, 10)
            transpose = window.transpose(0, 2, 1)
            for i in range(len(transpose)):
                jittering = augmentation_function(transpose[i], sigma)
                dictionary[pos_key].extend([jittering])
    return dictionary


def scatter_TSNE(tsne_proj, Y, alpha):
    """
       Description: Given an np_array outputted from the sklearn t-SNE function, and the corresponding
                                Y values, plot a scatter plot of the t-SNE points

       Args:
           tsne_proj -- np_array (Np array output from the TSNE function. Shape = (n_samples, n_components))
           Y -- np_array (Np array for the labels of the data orignially passed into the TSNE function)
           alpha -- double (Number that determines the transparancy level of the dots on the plot)

     """
    df_subset = {"negative": [], "positive": []}
    df_subset['positive'] = tsne_proj[:, 0]
    df_subset['negative'] = tsne_proj[:, 1]
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
