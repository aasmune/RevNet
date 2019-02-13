from numpy import genfromtxt, isnan, interp, zeros


def read_csv_files(filenames):
    data = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        filename = filenames[i]
        t = genfromtxt(filename, delimiter=',')         #try to optimize this later
        t = t[~isnan(t).any(axis=1)]
        data[i] = t
    
    return data


def create_single_table(data):
    n_samples = data[0].shape[0]
    n_channels = len(data)
    result = zeros(( n_samples, n_channels + 1))
    result[:,0] = data[0][:,0]

    for i in range(len(data)):
        result[:,i + 1] = interp(result[:,0], data[i][:,0], data[i][:,1])
    
    return result