from numpy import genfromtxt, isnan, interp, zeros
import pandas


def read_csv_files(filenames):
    data = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        t = pandas.read_csv(filenames[i], delimiter=',').values
        data[i] = t[~isnan(t).any(axis=1)]
    return data


def create_single_table(data):
    n_samples = data[0].shape[0]
    n_channels = len(data)
    result = zeros(( n_samples, n_channels))
    for i in range(n_channels):
        result[:,i] = interp(data[0][:,0], data[i][:,0], data[i][:,1])
    return result