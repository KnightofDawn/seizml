from record_data import rec_test_result
from matransfer import full_seizure_extract, gmi_dataset_extract
from get_root_dir import get_mat_root
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def main():
    # test_rectestresult()
    # test_fullseizextract()
    prob_tester()


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


def prob_tester():
    # Get Files
    ldir = get_mat_root() + "mlv2/threshbin/"
    gmitype = 'gmi5'  # use quantile-based MI threshold
    winsize = '2'
    state_switch = 's1'  # select the seizure state
    # randomState = 42  # fix random state
    np.random.seed(42)  # fix randomness
    th = 90  # select the MI threshold
    # SVC
    # kern = 'linear'
    # clf = SVC(kernel=kern, probability=True)
    # Logistic Regression
    pen = 'l2'
    clf = LogisticRegression(penalty=pen)
    # Random Forest
    # numEsts = 100
    # clf = RandomForestClassifier(n_estimators=numEsts)
    # AdaBoost
    # algo = "SAMME"
    # numEsts = 100
    # base_clf = DecisionTreeClassifier(max_depth=1)
    # clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=numEsts, algorithm=algo)

    # Train model
    X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, winsize, th, state_switch, interTestSize=0.66)
    clf.fit(X_train, y_train)


main()
