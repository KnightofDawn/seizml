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
state_switch = 's1'  # select the seizure state
maxWindows = 9
# randomState = 42  # fix random state
np.random.seed(42)  # fix randomness
th = 93
# SVC
# kern = 'linear'
# clf = SVC(kernel=kern, probability=True)
# Logistic Regression
# pen = 'l1'
# clf = LogisticRegression(penalty=pen)
# Random Forest
numEsts = 200
clf = RandomForestClassifier(n_estimators=numEsts)
# AdaBoost
# algo = "SAMME"
# numEsts = 100
# base_clf = DecisionTreeClassifier(max_depth=1)
# clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=numEsts, algorithm=algo)

# Train model
X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, winsize, th, state_switch,
                                                       interTestSize=0.66, max_windows=maxWindows)
clf.fit(X_train, y_train)

comment = dict()
comment['ClassifierType'] = clf.__class__.__name__
if "RandomForest" in comment['ClassifierType']:
    comment['number_estimators'] = numEsts
elif "AdaBoost" in comment['ClassifierType']:
    comment['BaseClassifier'] = base_clf.__class__.__name__
    comment['number_estimators'] = numEsts
    comment['algorithm'] = algo
elif "LogisticRegression" in comment['ClassifierType']:
    comment['penalty'] = pen
    comment['ClassifierType'] += pen.capitalize()
elif "SVC" in comment['ClassifierType']:
    comment['kernel'] = kern

comment['StateSwitch'] = state_switch

# Get detection data
# seiz = {'DV': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
#         'GB': ['3', '4'], 'JY': ['8', '9', '10', '11', '12', '13'], 'PE': ['2'], 'RS': ['3'], 'SW': ['2']}
seiz = {'PE': ['3']}
sdir = get_mat_root() + 'mlv2/fullseiz/pred/rf/_200/'
ldir = get_mat_root() + 'mlv2/fullseiz/th/'
for patient, seizure in seiz.iteritems():
    for s in seizure:
        print('Patient: {}, Seizure: {}'.format(patient, s))
        fname = '{}{}_S{}_fsmi_th{}.mat'.format(ldir, patient, s, th)
        fv, wind = full_seizure_extract(fname)
        probas_ = clf.predict_proba(fv)
        sfname = '{}{}_S{}_fsmi_th{}_{}_{}_.mat'.format(sdir, patient, s, th, state_switch, comment['ClassifierType'])
        classes = clf.classes_
        full_seizure_detect_save(sfname, probas_, classes, wind)
