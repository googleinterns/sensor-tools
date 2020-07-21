import numpy as np
from skimage.util.shape import view_as_windows

def create_train_test_validation(dictionary, pos_key, neg_key, percent_train, percent_test, percent_validation):
    """
        Description: creates a training, testing, and validation dictionary for binary data  

        Args:
            dictionary -- dict (Dictionary of data sorted into pos_key and neg_key: 
                                dictionary = {pos_key: [data, data,...], neg_key: [data, data, ...]})
            pos_key -- String (Key for the positive data samples in dictionary)
            neg_key -- String (Key for the negative data samples in dictionary)
            percent_train -- double ([0-1] Percentage of data to go to training set)
            percent_test -- double (0-1] Percentage of data to go to testing set)
            percent_validation -- double ([0-1] Percentage of data to go to validation set)

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