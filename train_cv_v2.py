import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy import interp
from matransfer import gmi_dataset_extract
from record_data import rec_test_result
from get_root_dir import get_mat_root
from joblib import Parallel, delayed
import multiprocessing


# ldir = get_mat_root() + "mlv2/threshbin/"
# gmitype = 'gmi5'
# winsize = '2'
# state_switch = 's1'
# inter_test_size = 0.66
# max_windows = 9
# np.random.seed(42)  # fix randomness
# SVC
# kern = 'linear'
# clf = SVC(kernel=kern, probability=True)
# Logistic Regression
# pen = 'l2'
# clf = LogisticRegression(penalty=pen)
# Random Forest
# numEsts = 100
# clf = RandomForestClassifier(n_estimators=numEsts)
# AdaBoost
# algo = "SAMME"
# numEsts = 100
# base_clf = DecisionTreeClassifier(max_depth=1)
# clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=numEsts, algorithm=algo)
N_e = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
num_cores = multiprocessing.cpu_count()

# thvals = range(1, 101)
# for th in thvals:  # for CVing over a range of MI Thresholds

def my_process(numEsts):
    th = 95 # for a fixed MI threshold
    ldir = get_mat_root() + "mlv2/threshbin/"
    gmitype = 'gmi5'
    winsize = '2'
    state_switch = 's2'
    inter_test_size = 0.66
    max_windows = 8
    np.random.seed(42)  # fix randomness
    clf = RandomForestClassifier(n_estimators=numEsts, n_jobs=-1)
    # print(clf.__class__.__name__ + " Threshold %0.0d/100" % th)
    # get seizure epochs from each patient with predetermined training/testing division
    X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, winsize, th, state_switch, inter_test_size, max_windows)

    cv = StratifiedKFold(y_train, n_folds=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = clf.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    score = auc(mean_fpr, mean_tpr)
    # plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % score, lw=2)

    # PLOTTING
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    # score = cross_val_score(clf, X_train, y_train.ravel(), cv=5)
    # print("Accuracy: %0.4f (+/- %0.4f)" % (100*score.mean(), 100*score.std() * 2))
    # print(score)
    # scores.append(score)
    # errs.append(score.std())

    savedir = "mlv2/rocbin/num_est/th{}/".format(th)
    savetype = "CV"
    comment = dict()
    comment['ClassifierType'] = clf.__class__.__name__
    comment['MaxWindows'] = float(max_windows % 5)
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
    appendString = ''.join([state_switch, str(numEsts)])
    rec_test_result(savetype, savedir, {'comment': comment, 'score': score, 'th': th}, appendString)

# maxscores = np.argmax(scores)
#
# plt.plot(range(1,101), scores, 'o')
# plt.plot(maxscores, scores[maxscores], 'ro')
# plt.axis([0, 100, 0.9, 1.])
# plt.xlabel("Threshold")
# plt.ylabel("CV AUC")
# plt.title("CV using " + state_switch.capitalize())
# plt.show()


Parallel(n_jobs=num_cores)(delayed(my_process)(i) for i in N_e)
