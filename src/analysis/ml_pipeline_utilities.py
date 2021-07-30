import numpy as np
import random
import copy

# Adding Hardcoding of the Geometric Plotting Order, Identity, and Axis Location
# Order to Plot (Probe Channel Name)
all_chan_map = {'z007': [2, 1, 10, 9, 18, 17, 4, 3, 12, 11, 20, 19, 5, 6, 13, 14, 21,
                         22, 7, 8, 15, 16, 23, 24, 29, 30, 31, 32, 25, 26, 27, 28],
                'z020': [6, 5, 9, 15, 3, 4, 10, 16, 1, 7, 13, 14, 2, 8, 12, 11],
                'z017': [1, 7, 13, 14, 3, 4, 10, 16, 2, 8, 12, 11, 6, 5, 9, 15]
                }

# Order to Plot on a 4x4 (OpenEphys Designation)
all_plot_maps = {'z007': [1, 0, 9, 8, 17, 16, 3, 2, 11, 10, 19, 18, 4, 5, 12, 13, 20,
                          21, 6, 7, 14, 15, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27],
                 'z020': [13, 12, 6, 5, 11, 15, 1, 7, 8, 14, 0, 4, 10, 9, 3, 2],
                 'z017': [8, 14, 0, 4, 11, 15, 1, 7, 10, 9, 3, 2, 13, 12, 6, 5]
                 }

# Location of Axis in plotting order
all_axis_orders = {'z007': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33],
                   'z020': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                   'z017': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                   }
# Channels to be Excluded from all Analysis
all_bad_channels = {'z007': [24, 28],
                    'z020': [1],
                    'z017': []
                    }
# Bird's Default Class Labels
all_label_instructions = {'z007': [1, 2, 3, 4, 5, 'I'],
                          'z020': [1, 2, 3, 4, 'I'],
                          'z017': [1, 2, 3, 4, 5, 6, 7]
                          }
# Bird's Default drop temps
all_drop_temps = {'z007': [6],
                  'z020': [5],
                  'z017': [7]}


def balance_classes(neural_data, safe=True, seed=False):
    """ Takes a List of Instances of the Time Series and Balances out all classes to be equal size
    (Approach 1: All Classes Set to be Equal)

    Parameters
    ----------
    neural_data : list | (classes, instances, channels, samples)
        Neural Data to be used in PCA-PSD Analysis
    safe : bool
        If True the function will make a soft copy of the neural data to prevent changes to the original input,
        if false the original data is altered instead
    seed : int, optional
        sets the seed for the pseudo-random module, defaults to not setting the seed.

    Returns
    -------
    balanced_data : list | (classes, instances, channels, samples)
        Randomly Rebalanced Neural Data to be used in PCA-PSD Analysis (All Sets are equal length)
    safe : bool, optional
        If True a deepcopy is made of the neural_data internally and returned.

    """

    if safe:
        balanced_data = copy.deepcopy(neural_data)  # Deep Copy
    else:
        balanced_data = neural_data  # Not a Shallow Copy or Deep Copy (Assignment)
    group_sizes = [len(events) for events in neural_data]  # Number of Instances per Class

    minimum = min(np.unique(group_sizes))  # Size of Smallest Class
    focus_mask = [index for index, value in enumerate(group_sizes) if value > minimum]  # Index of Larger Classes

    for needs_help in focus_mask:
        big = len(neural_data[needs_help])
        if seed:
            random.seed(seed)
        selected = random.sample(range(0, big), minimum, )  # Select the instances to Use
        balanced_data[needs_help] = neural_data[needs_help][selected]  # Reduce Instances to Those Selected

    return balanced_data


def get_priors(num_labels):
    priors = np.zeros((num_labels,))
    priors[:] = 1 / num_labels
    return priors
