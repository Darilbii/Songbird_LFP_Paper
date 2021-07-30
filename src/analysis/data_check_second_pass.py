import numpy as np

# Second Pass Epoch Check

def range_overlapping(x, y):
    if x.start == x.stop or y.start == y.stop:
        return False
    return ((x.start < y.stop  and x.stop > y.start) or
            (x.stop  > y.start and y.stop > x.start))


def overlap(x, y):
    if not range_overlapping(x, y):
        return set()
    return set(range(max(x.start, y.start), min(x.stop, y.stop ) +1))


def check_epoch_overlap(epoch_times, epoch_index: list):
    # Determine if any of the Epoch Overlap in time

    overlapping_epoch = []
    for index, i in enumerate(epoch_index):
        focus_range = range(int(epoch_times[i, 0]), int(epoch_times[i, 1] + 1), 1)
        for j in epoch_index[index + 1:]:
            #         print(j)
            test_range = range(int(epoch_times[j, 0]), int(epoch_times[j, 1] + 1), 1)
            if range_overlapping(focus_range, test_range):
                print(f"{i} overlaps with {j}")
                overlapping_epoch.append([i, j])

    overlapping_epoch = np.asarray(overlapping_epoch)

    return overlapping_epoch


def quantify_overlap(epoch_times, overlapping_epoch, verbose=False):
    # What is the Duration of the Overlap

    length_of_overlap = []

    for first, last in overlapping_epoch:
        first_range = range(int(epoch_times[first, 0]), int(epoch_times[first, 1] + 1), 1)
        last_range = range(int(epoch_times[last, 0]), int(epoch_times[last, 1] + 1), 1)

        epoch_overlap = len(overlap(first_range, last_range))
        if verbose:
            print(epoch_overlap / 30000)
        length_of_overlap.append(epoch_overlap)
    length_of_overlap = np.asarray(length_of_overlap)

    return length_of_overlap


