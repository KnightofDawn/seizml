from record_data import rec_test_result
from matransfer import full_seizure_extract
from get_root_dir import get_mat_root
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def main():
    # test_rectestresult()
    test_fullseizextract()


def test_rectestresult():
    clf = DecisionTreeClassifier(max_depth=2)
    scores = np.random.rand(10, 1)
    savedir = "mlv2/rocbin/"
    savetype = "CV"
    comment = vars(clf)
    comment['ClassifierType'] = clf.__class__.__name__
    subdict = {'fld1': None, 'fld2': [1, 2, 3]}
    maindict = {'sub': subdict, 'mfld1': [3, 2, 1], 'mfld2': None}
    # comment = dict((k, v) for k, v in comment.iteritems() if v is not None)
    rec_test_result(savetype, savedir, {'comment': comment, 'scores': scores, 'testd': maindict})
    # end result is verified by looking at the saved .mat file in the chosen directory


def test_fullseizextract():
    ldir = get_mat_root() + 'mlv2/fullseiz/th/'
    patient = 'DV'
    seizure = '10'
    mf = full_seizure_extract(ldir, patient, seizure, 90)

    print('dummy line')


main()
