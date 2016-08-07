import numpy as np
from imblearn.ensemble import EasyEnsemble


# Return the probability of being the positive class given an EasyEnsemble of 'nsubs' ensembles
def easy_ensemble_classifier(clf, x_train, y_train, x_test, nsubs, repl):
    ee = EasyEnsemble(n_subsets=nsubs, replacement=repl)  # Create EasyEnsemble object
    X_train_res, y_train_res = ee.fit_sample(x_train, y_train)  # re-sample the data
    clfs = []
    i = 0
    preds_ = np.zeros([1, np.shape(x_test)[0]])

    # Iterate through sub-samples:
    for xtrain in X_train_res:
        clfs += [clf]
        clfs[i].fit(xtrain, y_train_res[i])
        preds_ = np.add(preds_, clfs[i].predict(x_test))
        i += 1

    return np.divide(preds_, nsubs)


