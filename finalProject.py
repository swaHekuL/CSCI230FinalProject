# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:52:23 2022

@author: luke Haws, Jared Cordova, Dario Fumarola, Ryan Messick
"""

# import statements
from eda import returnData
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.base import clone
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.model_selection import validation_curve




class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def metrics(model, X_val, y_val, X_train, y_train, X_test, y_test):
   y_pred = model.predict(X_test)
   
   # k-fold cross validation
   print('\n10-fold Cross Validation for', str(model), ':')
   kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

   accScores = []
   preScores = []
   recScores = []
   f1Scores = []

   for k, (train, test) in enumerate(kfold):
       model.fit(X_train[train], y_train[train])
       y_pred_cv = model.predict(X_train[test])
       accScore = model.score(X_train[test], y_train[test])
       preScore = precision_score(y_true=y_train[test], y_pred=y_pred_cv, average = 'weighted')
       recScore = recall_score(y_true = y_train[test],y_pred = y_pred_cv, average = 'weighted')
       f1Score = f1_score(y_true = y_train[test], y_pred = y_pred_cv, average = 'weighted')
       accScores.append(accScore)
       preScores.append(preScore)
       recScores.append(recScore)
       f1Scores.append(f1Score)
       print('Fold: %2d, Acc: %.3f Pre: %.3f Rec: %.3f f1: %.3f' % (k+1, accScore, preScore, recScore, f1Score))
   print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(accScores), np.std(accScores)))
   print('CV precision: %.3f +/- %.3f' % (np.mean(preScores), np.std(preScores)))
   print('CV recall: %.3f +/- %.3f' % (np.mean(recScores), np.std(recScores)))
   print('CV f1: %.3f +/- %.3f' % (np.mean(f1Scores), np.std(f1Scores)))
   
   # confusion matrix
   confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels = [1,0])
   print('\nConfusion Matrix:\n', confmat)
   
   '''
   # # ROC
   # pipe_svm = make_pipeline(PCA(n_components = 2),
   #                          SVC(C = 10.0,
   #                                    kernel = 'rbf',
   #                                    # degree = 3,
   #                                    gamma = 'auto',
   #                                    coef0 = 0,
   #                                    shrinking = True,
   #                                    probability = True,
   #                                    tol = 0.000001,
   #                                    cache_size = 300,
   #                                    class_weight = None,
   #                                    verbose = False,
   #                                    max_iter = -1,
   #                                    decision_function_shape = 'ovr',
   #                                    break_ties = False,
   #                                    random_state = 1
   #                                    )
   #                          )
   
   # cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
   
   # fig = plt.figure(figsize=(7, 5))
   
   # mean_tpr = 0.0
   # mean_fpr = np.linspace(0, 1, 100)
   # all_tpr = []
   
   # for i, (train, test) in enumerate(cv):
   #     probas = pipe_svm.fit(X_train[train],
   #                          y_train[train]).predict_proba(X_train[test])

   #     fpr, tpr, thresholds = roc_curve(y_train[test],
   #                                      probas[:, 1],
   #                                      pos_label=1)
   #     mean_tpr += interp(mean_fpr, fpr, tpr)
   #     mean_tpr[0] = 0.0
   #     roc_auc = auc(fpr, tpr)
   #     plt.plot(fpr,
   #              tpr,
   #              label='ROC fold %d (area = %0.2f)'
   #                    % (i+1, roc_auc))

   # plt.plot([0, 1],
   #          [0, 1],
   #          linestyle='--',
   #          color=(0.6, 0.6, 0.6),
   #          label='Random guessing')

   # mean_tpr /= len(cv)
   # mean_tpr[-1] = 1.0
   # mean_auc = auc(mean_fpr, mean_tpr)
   # plt.plot(mean_fpr, mean_tpr, 'k--',
   #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
   # plt.plot([0, 0, 1],
   #          [0, 1, 1],
   #          linestyle=':',
   #          color='black',
   #          label='Perfect performance')

   # plt.xlim([-0.05, 1.05])
   # plt.ylim([-0.05, 1.05])
   # plt.xlabel('False positive rate')
   # plt.ylabel('True positive rate')
   # plt.legend(loc="lower right")

   # plt.tight_layout()
   # plt.show()
   '''
   
def SBS_function(model, X_val, y_val):
   
   # selecting features
   sbs = SBS(model, k_features = 1)
   sbs.fit(X_val, y_val)
   
   # plotting performance of feature subsets
   k_feat = [len(k) for k in sbs.subsets_]

   plt.plot(k_feat, sbs.scores_, marker='o')
   plt.ylim([0.5, 1.02])
   plt.ylabel('Accuracy')
   plt.xlabel('Number of features')
   plt.grid()
   plt.tight_layout()
   # plt.savefig('images/04_08.png', dpi=300)
   plt.show()
   
def gridSearch(model, X_val, y_val, X_train, y_train, param_grid):
   
   gs = GridSearchCV(estimator = model(),
                     param_grid = param_grid,
                     scoring = 'accuracy',
                     cv=2)

   scores = cross_val_score(gs, X_val, y_val, 
                            scoring='accuracy', cv=5)
   
   print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                         np.std(scores)))
   
   gs = gs.fit(X_train, y_train)
   print("Best Accuracy: %.3f" % gs.best_score_)
   print(gs.best_params_)

def main():
   
   # the data
   X, y = returnData()
   
   X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=1, stratify = y)

   X_test, X_val, y_test, y_val = train_test_split(
      X_test, y_test, test_size=0.5, random_state=1, stratify=y_test)
   
   # SMOTE this yields equal values for each output, so thats good!
   sm = SMOTE(sampling_strategy='not majority',
              random_state = 1,
              n_jobs = 2)
   
   X_train, y_train = sm.fit_resample(X_train, y_train)
   
   X_train, y_train = np.array(X_train), np.array(y_train).ravel()
   X_test, y_test = np.array(X_test), np.array(y_test).ravel()
   X_val, y_val = np.array(X_val), np.array(y_val).ravel()

   ####################### SVM SECTION #######################
   print("SVM Section\n")
   # SVC model
   svm = SVC(C = 100.0,
             kernel = 'rbf',
             # degree = 3,
             gamma = 'scale',
             coef0 = 0,
             shrinking = True,
             probability = False,
             tol = 0.001,
             cache_size = 300,
             class_weight = None,
             verbose = False,
             max_iter = -1,
             decision_function_shape = 'ovr',
             break_ties = False,
             random_state = 1
             )
   
   svm.fit(X_train, y_train)
   
   metrics(svm, X_val, y_val, X_train, y_train, X_test, y_test)
   
   #SBS_function(svm, X_val, y_val)
   
   # svc_param_grid = [{'C': [100],
   #                'kernel': ['rbf'],
   #                #'degree': [3], get ignored if using rbf
   #                'gamma': ['scale'],
   #                'coef0': [0],
   #                'shrinking': [True],
   #                'probability': [False],
   #                'tol': [0.001, 0.01],
   #                'cache_size': [200], # don't think we need to change this
   #                'class_weight': [None],
   #                'verbose': [False],
   #                'max_iter': [-1],
   #                'decision_function_shape': ['ovr'],
   #                'break_ties': [False],
   #                'random_state': [1]
   #                }]
   
   #gridSearch(SVC, X_val, y_val, X_train, y_train, svc_param_grid)
   
   ####################### DECISION TREE SECTION #######################
   print("\nDecision Tree Section\n")
   tree_model = DecisionTreeClassifier(criterion='entropy',
                                        splitter = 'best',
                                        max_depth = 45, 
                                        min_samples_split = 4,
                                        min_samples_leaf = 1,
                                        min_weight_fraction_leaf = 0,
                                        max_features = 6,
                                        random_state = 1,
                                        max_leaf_nodes = None,
                                        min_impurity_decrease = 0,
                                        class_weight = None,
                                        ccp_alpha = 0)
   tree_model.fit(X_train, y_train)
   
   '''
   # SBS
   # selecting features
   sbs = SBS(tree_model, k_features = 1)
   sbs.fit(X_val, y_val)
  
   # plotting performance of feature subsets
   k_feat = [len(k) for k in sbs.subsets_]
   plt.plot(k_feat, sbs.scores_, marker='o')
   plt.ylim([0.5, 1.02])
   plt.ylabel('Accuracy')
   plt.xlabel('Number of features')
   plt.grid()
   plt.tight_layout()
   # plt.savefig('images/04_08.png', dpi=300)
   plt.show()
   '''
   
   y_pred = tree_model.predict(X_test)
   
   tree.plot_tree(tree_model)
   plt.show()
   
   dot_data = export_graphviz(tree_model,
                              filled=True, 
                              rounded=True,
                              class_names = ['1','0'],
                              feature_names = ['Years Out', 
                                      'Spouse a Grad',
                                      'Academic Activities',
                                      'Activities',
                                      'Varsity Athletics',
                                      'All American',
                                      'Honor Societies',
                                      'UG Academic Honors',
                                      'Alumni Admissions Program',
                                      'Alumni Board',
                                      'Chapter Volunteers',
                                      'Reunion Class Committee',
                                      'Business',
                                      'Arts',
                                      'Humanities',
                                      'SocialSciences',
                                      'STEM', 
                                      'OtherMajor',
                                      'Legal_Services',
                                      'Investments',
                                      'Real_Estate',
                                      'Education',
                                      'Healthcare',
                                      'Government',
                                      'Retail',
                                      'Public_Relations',
                                      'Utilities',
                                      'Tech',
                                      'Entertainment',
                                      'Other'],
                                      out_file=None) 
   graph = graph_from_dot_data(dot_data) 
   graph.write_png('tree.png') 
    
# metrics  
 
   # k-fold cross validation
   print('\n10-fold Cross Validation:')
   kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
   accScores = []
   preScores = []
   recScores = []
   f1Scores = []

   for k, (train, test) in enumerate(kfold):
        tree_model.fit(X_train[train], y_train[train])
        y_pred_cv = tree_model.predict(X_train[test])
        accScore = tree_model.score(X_train[test], y_train[test])
        preScore = precision_score(y_true=y_train[test], y_pred=y_pred_cv, average = 'weighted')
        recScore = recall_score(y_true = y_train[test],y_pred = y_pred_cv, average = 'weighted')
        f1Score = f1_score(y_true = y_train[test], y_pred = y_pred_cv, average = 'weighted')
        accScores.append(accScore)
        preScores.append(preScore)
        recScores.append(recScore)
        f1Scores.append(f1Score)
        print('Fold: %2d, Acc: %.3f Pre: %.3f Rec: %.3f f1: %.3f' % (k+1, accScore, preScore, recScore, f1Score))
   print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(accScores), np.std(accScores)))
   print('CV precision: %.3f +/- %.3f' % (np.mean(preScores), np.std(preScores)))
   print('CV recall: %.3f +/- %.3f' % (np.mean(recScores), np.std(recScores)))
   print('CV f1: %.3f +/- %.3f' % (np.mean(f1Scores), np.std(f1Scores)))
   
   # confusion matrix
   confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels = [1,0])
   print('\nConfusion Matrix:\n', confmat)

   
   # # grid search
   # param_grid = [{'criterion': ['entropy'],
   #                'splitter': ['best'],
   #                'max_depth': [45],
   #                'min_samples_split': [4],
   #                'min_samples_leaf': [1],
   #                'min_weight_fraction_leaf': [0],
   #                'max_features': ['auto'],
   #                # skip random state
   #                'max_leaf_nodes': [None], # don't wanna mess w this
   #                'min_impurity_decrease': [0],
   #                # skip class weight
   #                'ccp_alpha': [0]}]
   
   # gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 1),
   #                   param_grid = param_grid,
   #                   scoring = 'accuracy',
   #                   cv = 2)
   
   # scores = cross_val_score(gs, X_val, y_val, 
   #                          scoring='accuracy', cv=5)
   # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), 
   #                                       np.std(scores)))
   # gs = gs.fit(X_train, y_train)
   # print(gs.best_score_)
   # print(gs.best_params_)
   

  #  # validation curve for max depth
  #  param_range = [43,44,45,46,47]
  #  train_scores, test_scores = validation_curve(
  #                  estimator=tree_model, 
  #                  X=X_train, 
  #                  y=y_train, 
  #                  param_name='max_depth', 
  #                  param_range=param_range,
  #                  cv=10)
   
  #  train_mean = np.mean(train_scores, axis=1)
  #  train_std = np.std(train_scores, axis=1)
  #  test_mean = np.mean(test_scores, axis=1)
  #  test_std = np.std(test_scores, axis=1)

  #  plt.plot(param_range, train_mean, 
  #            color='blue', marker='o', 
  #            markersize=5, label='Training accuracy')

  #  plt.fill_between(param_range, train_mean + train_std,
  #                    train_mean - train_std, alpha=0.15,
  #                    color='blue')

  #  plt.plot(param_range, test_mean, 
  #            color='green', linestyle='--', 
  #            marker='s', markersize=5, 
  #            label='Validation accuracy')

  #  plt.fill_between(param_range, 
  #                    test_mean + test_std,
  #                    test_mean - test_std, 
  #                    alpha=0.15, color='green')

  #  plt.grid()
  #  plt.xscale('log')
  #  plt.legend(loc='lower right')
  #  plt.xlabel('Max Depth')
  #  plt.ylabel('Accuracy')
  #  plt.ylim([0.8, 1.0])
  #  plt.tight_layout()
  # # plt.savefig('images/06_06.png', dpi=300)
  #  plt.show()
  
  ####################### RANDOM FOREST SECTION #######################
   print("\nRandom Forest Section\n")
   # random forest
   
   y_train = y_train.ravel()
   
   forest = RandomForestClassifier(criterion='gini',
                                   n_estimators=25, 
                                   random_state=1,
                                   n_jobs=2)
   
   forest.fit(X_train, y_train)
   y_pred = forest.predict(X_test)
  
   accScores = []
   preScores = []
   recScores = []
   f1Scores = []
  
   # k-fold cross validation
   print('\n10-fold Cross Validation:')
   kfold2 = StratifiedKFold(n_splits=10).split(X_train, y_train)
   
   for k, (train, test) in enumerate(kfold2):
       forest.fit(X_train[train], y_train[train])
       y_pred_cv = forest.predict(X_train[test])
       accScore = forest.score(X_train[test], y_train[test])
       preScore = precision_score(y_true=y_train[test], y_pred=y_pred_cv, average = 'weighted')
       recScore = recall_score(y_true = y_train[test],y_pred = y_pred_cv, average = 'weighted')
       f1Score = f1_score(y_true = y_train[test], y_pred = y_pred_cv, average = 'weighted')
       accScores.append(accScore)
       preScores.append(preScore)
       recScores.append(recScore)
       f1Scores.append(f1Score)
       print('Fold: %2d, Acc: %.3f Pre: %.3f Rec: %.3f f1: %.3f' % (k+1, accScore, preScore, recScore, f1Score))
   print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(accScores), np.std(accScores)))
   print('CV precision: %.3f +/- %.3f' % (np.mean(preScores), np.std(preScores)))
   print('CV recall: %.3f +/- %.3f' % (np.mean(recScores), np.std(recScores)))
   print('CV f1: %.3f +/- %.3f' % (np.mean(f1Scores), np.std(f1Scores)))
   
   # confusion matrix
   confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels = [1,0])
   print('\nConfusion Matrix:\n', confmat)
   
if __name__ == "__main__":
   main()