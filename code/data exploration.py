# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:40:13 2019

@author: lihan
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)



training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

training_data.dtypes
training_data.empty
training_data = training_data.drop(["ID_code"],axis=1)

training_data["target"].value_counts()
sns.countplot(x="target",data=training_data)
Y = training_data["target"]
X = training_data.drop(["target"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
import time
start_time = time.time()
logreg.fit(X_train, y_train)
print("The logistic regression takes %s seconds ---" % (time.time() - start_time))
from sklearn.metrics import roc_auc_score
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

# try to implement lgb method
import lightgbm as lgb
features = [c for c in training_data.columns if c not in ['ID_code', 'target']]
target = training_data['target']

param = {
        'num_leaves': 6,
        'max_bin': 63,
        'min_data_in_leaf': 45,
        'learning_rate': 0.01,
        'min_sum_hessian_in_leaf': 0.000446,
        'bagging_fraction': 0.55, 
        'bagging_freq': 5, 
        'max_depth': 14,
        'save_binary': True,
        'seed': 31452,
        'feature_fraction_seed': 31415,
         'feature_fraction': 0.51,
        'bagging_seed': 31415,
        'drop_seed': 31415,
        'data_random_seed': 31415,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }

folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=31415)
oof = np.zeros(len(training_data))
predictions = np.zeros(len(test_data))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(training_data.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(training_data.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(training_data.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 15000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 250)
    oof[val_idx] = clf.predict(training_data.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_data[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub_df = pd.DataFrame({"ID_code":test_data["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)