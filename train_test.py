import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from matransfer import gmi_dataset_extract
from get_root_dir import get_mat_root
from sklearn.ensemble import VotingClassifier
from imblearn.ensemble import EasyEnsemble


ldir = get_mat_root() + "mlv2/threshbin/"
gmitype = 'gmi5'  # use quantile-based MI threshold
state_switch = 's2'  # select the seizure state
randomState = 42  # fix random state
th = 80  # select the MI threshold
numSubsets = 10

X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, th, state_switch, randomState, interTestSize=0.66)

# clf = BaggingClassifier(svm.SVC(kernel='linear', probability=True, random_state=randomState),
#                         n_estimators=20, random_state=randomState, max_samples=0.1)
# clf = svm.SVC(kernel='linear', probability=True, random_state=randomState)
ee = EasyEnsemble(random_state=randomState, n_subsets=numSubsets)
X_train_res, y_train_res = ee.fit_sample(X_train, y_train)
base_clf = svm.SVC(kernel='linear', probability=True, random_state=randomState)
clfs = []

for i, xtrain in enumerate(X_train_res):
    # print(y_train_res[i])
    clfs += ('sv%0.0d' % i, base_clf.fit(xtrain, y_train_res[i]))



probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC Test (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for S2')
plt.legend(loc="lower right")
plt.show()
