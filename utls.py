
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import gc


def cross_val(features, labels, model_type = 'forest', n_estimators = 100, print_feature_importances = False, C = 50, n_folds = 5):
 
    print('Training Data Shape: ', features.shape)

    features = np.array(features)

    k_fold = StratifiedKFold(n_splits = n_folds, shuffle = True)

    out_of_fold = np.zeros(features.shape[0])
    if print_feature_importances:
      print('\nFeature importances:\n')

    for train_indices, valid_indices in k_fold.split(features, labels):
        
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        if model_type == 'forest':
          model = RandomForestClassifier(n_estimators=n_estimators, criterion = 'entropy', max_features = None, n_jobs = -1)
        else:
          model = SVC(C=C, probability=True)

        model.fit(train_features, train_labels)

        if print_feature_importances:
          print(model.feature_importances_)

        out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, 1]
        
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    f1 = f1_score(labels, (out_of_fold>0.5))
    print('\nF1 Score: {:4}'.format(f1))

    return out_of_fold