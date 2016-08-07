import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


# collect interictal, thresholded MI epochs from the interictal data structures
def getinterictal(fname):
    matfile = sio.loadmat(fname) # load mat file
    fv = matfile['data'][0, 0][0] # initialize first row
    for i in range(1, len(matfile['data'][0])): # collect remaining rows
        # print(np.shape(fv))
        # print(np.shape(matfile['data'][0, i][0]))

        fv = np.vstack((fv, matfile['data'][0, i][0])) # use vstack to vertically concatenate
    return fv


# collect thresholded MI epochs from a seizure
def getseizure(fname):
    matfile = sio.loadmat(fname)  # load mat file
    fv = matfile['data'][0, 0][2]  # initialize first row
    fl = matfile['data'][0, 0][0]
    for i in range(1, len(matfile['data'][0])):  # collect remaining rows
        # print(np.shape(fv))
        # print(np.shape(matfile['data'][0, i][0]))

        fv = np.vstack((fv, matfile['data'][0, i][2]))  # use vstack to vertically concatenate
        fl = np.vstack((fl, matfile['data'][0, i][0]))
    return fv, fl


# Compile the training and test set from the patient seizure and interictal data. Returns training and test data/labels;
#   The training data/labels are shuffled.
def gmi_dataset_extract(ldir, gmiType, winSize, threshold, stateSwitch, interTestSize):  # random goes 2nd-to-last
    patients = ['DV', 'GB', 'SW', 'PE', 'RS', 'JY']
    test_rng = {'DV': [20, 27], 'GB': [4, 7], 'SW': [2, 3], 'PE': [2, 3], 'RS': [4, 5], 'JY': [8, 13]}
    train_rng = {'DV': [0, 19], 'GB': [0, 3], 'SW': [0, 1], 'PE': [0, 1], 'RS': [0, 3], 'JY': [0, 7]}

    first = True
    for pt in patients:
        fname = pt + "19_EEG_" + winSize + "sec_" + gmiType + "_th=%0.0d.mat" % threshold
        fv, fl = getseizure(ldir + fname)
        if first:
            test_data = fv[test_rng[pt][0]:test_rng[pt][1] + 1]
            train_data = fv[train_rng[pt][0]:train_rng[pt][1] + 1]
            test_lbls = fl[test_rng[pt][0]:test_rng[pt][1] + 1]
            train_lbls = fl[train_rng[pt][0]:train_rng[pt][1] + 1]
            first = False
        else:
            test_data = np.vstack((test_data, fv[test_rng[pt][0]:test_rng[pt][1] + 1]))
            train_data = np.vstack((train_data, fv[train_rng[pt][0]:train_rng[pt][1] + 1]))
            test_lbls = np.vstack((test_lbls, fl[test_rng[pt][0]:test_rng[pt][1] + 1]))
            train_lbls = np.vstack((train_lbls, fl[train_rng[pt][0]:train_rng[pt][1] + 1]))

    # handle the state_switch
    if stateSwitch == "s1":
        test_data = np.squeeze(test_data[np.where(test_lbls.ravel() == 0), ])
        train_data = np.squeeze(train_data[np.where(train_lbls.ravel() == 0), ])
        test_lbls = np.ones([np.shape(test_data)[0], 1])
        train_lbls = np.ones([np.shape(train_data)[0], 1])
    elif stateSwitch == "s2":
        test_data = np.squeeze(test_data[np.where(test_lbls.ravel() == 1), ])
        train_data = np.squeeze(train_data[np.where(train_lbls.ravel() == 1), ])
        test_lbls = np.ones([np.shape(test_data)[0], 1])
        train_lbls = np.ones([np.shape(train_data)[0], 1])

    # get the interictal data
    fname = "6P19_EEG_10sec_" + gmiType + "_th=%0.0d.mat" % threshold
    fv = getinterictal(ldir + fname)
    fl = np.zeros([np.shape(fv)[0], 1])
    fv_train, fv_test, fl_train, fl_test = train_test_split(fv, fl, test_size=interTestSize)  #, random_state=randomState)

    # compile the final train and test set from both interictal and seizure states
    X_train = np.vstack((train_data, fv_train))
    X_test = np.vstack((test_data, fv_test))
    y_train = np.vstack((train_lbls, fl_train))
    y_test = np.vstack((test_lbls, fl_test))
    # print(np.shape(y_train) + np.shape(X_train))
    X_train, y_train = shuffle(X_train, y_train.ravel())  #, random_state=randomState)

    X_train = np.float_(X_train)
    X_test = np.float_(X_test)

    return X_train, y_train, X_test, y_test


