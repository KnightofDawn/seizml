import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import BernoulliNB
from matransfer import gmi_dataset_extract, full_seizure_extract, full_seizure_detect_save
from get_root_dir import get_mat_root
from record_data import rec_test_result


# Get Files
ldir = get_mat_root() + "mlv2/threshbin/"
gmitype = 'gmi5'  # use quantile-based MI threshold
winsize = '2'
state_switch = 's2'  # select the seizure state
# randomState = 42  # fix random state
np.random.seed(42)  # fix randomness
th = 90  # select the MI threshold
kern = 'linear'

# Train model
X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, winsize, th, state_switch, interTestSize=0.66)
clf = SVC(kernel=kern, probability=True)
clf.fit(X_train, y_train)

# Get detection data
seiz = {'DV': ['11', '12', '13', '14'], 'GB': ['3', '4'], 'JY': ['5', '6', '7'], 'PE': ['2'], 'RS': ['3'], 'SW': ['2']}
sdir = get_mat_root() + '/mlv2/fullseiz/pred/'
ldir = get_mat_root() + '/mlv2/fullseiz/th/'
for patient, seizure in seiz.iteritems():
    for s in seizure:
        print('Patient: {}, Seizure: {}'.format(patient, s))
        fname = '{}{}_S{}_fsmi_th{}.mat'.format(ldir, patient, s, th)
        fv, wind = full_seizure_extract(fname)
        probas_ = clf.predict_proba(fv)
        sfname = '{}{}_S{}_fsmi_th{}_{}_det_.mat'.format(sdir, patient, s, th, state_switch)
        classes = clf.classes_
        full_seizure_detect_save(sfname, probas_, classes, wind)
