import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import BernoulliNB
from matransfer import gmi_dataset_extract
from get_root_dir import get_mat_root
from record_data import rec_test_result


ldir = get_mat_root() + "mlv2/threshbin/"
gmitype = 'gmi5'  # use quantile-based MI threshold
winsize = '2'
state_switch = 's1s2'  # select the seizure state
# randomState = 42  # fix random state
np.random.seed(42)  # fix randomness
th = 50  # select the MI threshold
# SVC
# kern = 'linear'
# clf = SVC(kernel=kern, probability=True)
# Logistic Regression
# pen = 'l2'
# clf = LogisticRegression(penalty=pen)
# Random Forest
numEsts = 100
clf = RandomForestClassifier(n_estimators=numEsts)
# AdaBoost
# algo = "SAMME"
# numEsts = 100
# base_clf = DecisionTreeClassifier(max_depth=1)
# clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=numEsts, algorithm=algo)

X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, winsize, th, state_switch, interTestSize=0.66)
clf.fit(X_train, y_train)
probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)


savedir = "mlv2/rocbin/"
savetype = "TEST"
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
elif "SVC" in comment['ClassifierType']:
    comment['kernel'] = kern

comment['StateSwitch'] = state_switch
rec_test_result(savetype, savedir,
                {'comment': comment, 'fpr': fpr, 'th': th, 'tpr': tpr, 'rocth': thresholds, 'rocauc': roc_auc})


# print(roc_auc)

# plt.plot(fpr, tpr, lw=1, label='ROC Test (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC for %s, Number of Ensembles: %0.0d' % (state_switch, numest))
# plt.legend(loc="lower right")
# plt.show()
