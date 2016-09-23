import scipy.io as sio
import datetime
import sklearn
import numpy as np
import inspect
from get_root_dir import get_mat_root


# type    - type of result (i.e. CV, testing)
# loc     - where should the results be saved
# in_dict - dictionary object containing the fields to be saved
def rec_test_result(save_type, loc, in_dict):

    saveloc = get_mat_root() + loc
    ttl = save_type + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # ttl = save_type + '_' + datetime.datetime.now().strftime("%Y-%m-%d_") + append_string  # need append_string as input
    sv_dict = dict_cleaner(in_dict)
    # print('Saving .mat file...')
    sio.savemat(saveloc + ttl, sv_dict)
    # print('Saving .mat file... Done.')
    return None


# Recursive dictionary cleaning method to remove keys with None value
#  meant for scipy.io.savemat which fails when dictionaries contain such key/values
def dict_cleaner(dirty_dict):

    clean_dict = dict()
    for dk, dv in dirty_dict.iteritems():
        if isinstance(dv, dict):
            clean_dict[dk] = dict_cleaner(dv)
        else:
            clean_dict[dk] = dv

    return clean_dict
