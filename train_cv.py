import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from scipy import interp
from matransfer import gmi_dataset_extract

ldir = "/Users/hiltontj/Documents/MATLAB/pancreas/mlv2/threshbin/"
gmitype = 'gmi5'
state_switch = 's2'
randomState = 42

scores = []
errs = []
for th in range(1, 101):
    print("Threshold %0.0d/100" % th)
    # get seizure epochs from each patient with predetermined training/testing division
    X_train, y_train, X_test, y_test = gmi_dataset_extract(ldir, gmitype, th, state_switch, randomState, interTestSize=0.66)

    cv = StratifiedKFold(y_train, n_folds=5)
    clf = svm.SVC(kernel='linear', probability=True, random_state=randomState)

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
    print(score)
    scores.append(score)
    # errs.append(score.std())

savedir = "/Users/hiltontj/Documents/MATLAB/pancreas/mlv2/"
fname = "CV_" + gmitype + "_" + state_switch + ".txt"
np.savetxt(savedir + fname, np.c_[scores], delimiter=',')

maxscores = np.argmax(scores)

plt.plot(range(1,101), scores, 'o')
plt.plot(maxscores, scores[maxscores], 'ro')
plt.axis([0, 100, 0.9, 1.])
plt.xlabel("Threshold")
plt.ylabel("CV AUC")
plt.title("CV using " + state_switch.capitalize())
plt.show()

