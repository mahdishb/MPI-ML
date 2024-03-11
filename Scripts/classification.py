# k fold
from sklearn.model_selection import KFold
from collections import Counter

y_true= np.array([])
y_pred= np.array([])
y_pred_total = pd.DataFrame()

pr_kfold = []
re_kfold = []
f1_kfold = []
f1m_kfold = []
f1w_kfold = []
acc_kfold = []
auc_roc_kfold = []
auc_ap_kfold = []

kf= KFold(n_splits= 5, shuffle= True)

for train_index, test_index in kf.split(x_data_potneg):
    X_train , X_test = x_data_potneg.iloc[train_index], x_data_potneg.iloc[test_index]
    y_train , y_test = y_data_potneg.iloc[train_index] , y_data_potneg.iloc[test_index]

    clf = RandomForestClassifier(n_estimators= 100)

    rus= RandomUnderSampler()

    X_train_res, y_train_res= rus.fit_resample(X_train, y_train)

    # print('Original Set: 1-0 ', (y_train['medium'].value_counts()[1],y_train['medium'].value_counts()[0]))
    # print('Resampled Set: 1-0 ', (y_train_res['medium'].value_counts()[1],y_train_res['medium'].value_counts()[0]))

    clf.fit(X_train_res, y_train_res)

    # y_pred= np.hstack((y_pred, clf.predict(X_test)))
    # y_true= np.hstack((y_true, y_test))

    y_pred = clf.predict(X_test)

    # save the labels to get the most repotred metabolites in the true reactions

    df_temp = pd.DataFrame(y_pred)
    df_temp = df_temp.rename(columns={0: 'class_prediction'})
    df_temp['idx'] = X_test.index.values

    y_pred_total = pd.concat([y_pred_total, df_temp], ignore_index=True)

    prec = precision_score(y_test, y_pred)
    reca = recall_score(y_test, y_pred)
    f1sc = f1_score(y_test, y_pred)
    f1scm = f1_score(y_test, y_pred, average='macro')
    f1scw = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    ap = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])

    pr_kfold.append(prec)
    re_kfold.append(reca)
    f1_kfold.append(f1sc)
    f1m_kfold.append(f1scm)
    f1w_kfold.append(f1scw)
    acc_kfold.append(acc)
    auc_roc_kfold.append(roc)
    auc_ap_kfold.append(ap)


prec_mean = mean(pr_kfold)
reca_mean = mean(re_kfold)
f1sc_mean = mean(f1_kfold)
f1scm_mean = mean(f1m_kfold)
f1scw_mean = mean(f1w_kfold)
acc_mean = mean(acc_kfold)
roc_mean = mean(auc_roc_kfold)
ap_mean = mean(auc_ap_kfold)

results = results.append({'run_name': run+'_'+labeling, 'classifier': 'RF', 'precision': prec_mean, 'recall':reca_mean, 'f1-score':f1sc_mean, 'f1(macro)':f1scm_mean, 'f1(weighted)':f1scw_mean, 'accuracy':acc_mean, 'AUC(ROC)':roc_mean, 'AUC(PR)':ap_mean}, ignore_index = True)


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 3))
cnf = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax1)
roc = RocCurveDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax2)
pr = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax3)
plt.grid(False)
plt.tight_layout(pad=2.0)
# plt.show()
fig.savefig(f"{images_dir}/pot_rf_{run}_{labeling}.png")

# k fold
from sklearn.model_selection import KFold
from collections import Counter

y_true= np.array([])
y_pred= np.array([])

pr_kfold = []
re_kfold = []
f1_kfold = []
f1m_kfold = []
f1w_kfold = []
acc_kfold = []
auc_roc_kfold = []
auc_ap_kfold = []

kf= KFold(n_splits= 10, shuffle= True)

for train_index, test_index in kf.split(x_data_potneg):
    X_train , X_test = x_data_potneg.iloc[train_index], x_data_potneg.iloc[test_index]
    y_train , y_test = y_data_potneg.iloc[train_index] , y_data_potneg.iloc[test_index]

    parameters = {'C': 0.1, 'gamma': 10, 'kernel': 'rbf'}

    clf = SVC(**parameters)

    X_train_res, y_train_res= rus.fit_resample(X_train, y_train)

    # print('Original Set: 1-0 ', (y_train['medium'].value_counts()[1],y_train['medium'].value_counts()[0]))
    # print('Resampled Set: 1-0 ', (y_train_res['medium'].value_counts()[1],y_train_res['medium'].value_counts()[0]))

    clf.fit(X_train_res, y_train_res)

    # y_pred= np.hstack((y_pred, clf.predict(X_test)))
    # y_true= np.hstack((y_true, y_test))

    y_pred = clf.predict(X_test)


    prec = precision_score(y_test, y_pred)
    reca = recall_score(y_test, y_pred)
    f1sc = f1_score(y_test, y_pred)
    f1scm = f1_score(y_test, y_pred, average='macro')
    f1scw = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, clf.decision_function(X_test))
    ap = average_precision_score(y_test, clf.decision_function(X_test))

    pr_kfold.append(prec)
    re_kfold.append(reca)
    f1_kfold.append(f1sc)
    f1m_kfold.append(f1scm)
    f1w_kfold.append(f1scw)
    acc_kfold.append(acc)
    auc_roc_kfold.append(roc)
    auc_ap_kfold.append(ap)


prec_mean = mean(pr_kfold)
reca_mean = mean(re_kfold)
f1sc_mean = mean(f1_kfold)
f1scm_mean = mean(f1m_kfold)
f1scw_mean = mean(f1w_kfold)
acc_mean = mean(acc_kfold)
roc_mean = mean(auc_roc_kfold)
ap_mean = mean(auc_ap_kfold)

results = results.append({'run_name': run+'_'+labeling, 'classifier':'SVM', 'precision': prec_mean, 'recall':reca_mean, 'f1-score':f1sc_mean, 'f1(macro)':f1scm_mean, 'f1(weighted)':f1scw_mean, 'accuracy':acc_mean, 'AUC(ROC)':roc_mean, 'AUC(PR)':ap_mean}, ignore_index = True)


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 3))
cnf = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax1)
roc = RocCurveDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax2)
pr = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax3)
plt.grid(False)
plt.tight_layout(pad=2.0)
# plt.show()
fig.savefig(f"{images_dir}/pot_svm_{run}_{labeling}.png")

# k fold
from sklearn.model_selection import KFold
from collections import Counter

y_true= np.array([])
y_pred= np.array([])

pr_kfold = []
re_kfold = []
f1_kfold = []
f1m_kfold = []
f1w_kfold = []
acc_kfold = []
auc_roc_kfold = []
auc_ap_kfold = []

kf= KFold(n_splits= 5, shuffle= True)

for train_index, test_index in kf.split(x_data_potneg):
    X_train , X_test = x_data_potneg.iloc[train_index], x_data_potneg.iloc[test_index]
    y_train , y_test = y_data_potneg.iloc[train_index] , y_data_potneg.iloc[test_index]

    clf = MLPClassifier(random_state=1, max_iter=300)

    rus= RandomUnderSampler()

    X_train_res, y_train_res= rus.fit_resample(X_train, y_train)

    # print('Original Set: 1-0 ', (y_train['medium'].value_counts()[1],y_train['medium'].value_counts()[0]))
    # print('Resampled Set: 1-0 ', (y_train_res['medium'].value_counts()[1],y_train_res['medium'].value_counts()[0]))

    clf.fit(X_train_res, y_train_res)

    # y_pred= np.hstack((y_pred, clf.predict(X_test)))
    # y_true= np.hstack((y_true, y_test))

    y_pred = clf.predict(X_test)


    prec = precision_score(y_test, y_pred)
    reca = recall_score(y_test, y_pred)
    f1sc = f1_score(y_test, y_pred)
    f1scm = f1_score(y_test, y_pred, average='macro')
    f1scw = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    ap = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])

    pr_kfold.append(prec)
    re_kfold.append(reca)
    f1_kfold.append(f1sc)
    f1m_kfold.append(f1scm)
    f1w_kfold.append(f1scw)
    acc_kfold.append(acc)
    auc_roc_kfold.append(roc)
    auc_ap_kfold.append(ap)


prec_mean = mean(pr_kfold)
reca_mean = mean(re_kfold)
f1sc_mean = mean(f1_kfold)
f1scm_mean = mean(f1m_kfold)
f1scw_mean = mean(f1w_kfold)
acc_mean = mean(acc_kfold)
roc_mean = mean(auc_roc_kfold)
ap_mean = mean(auc_ap_kfold)

results = results.append({'run_name': run+'_'+labeling, 'classifier':'MLP', 'precision': prec_mean, 'recall':reca_mean, 'f1-score':f1sc_mean, 'f1(macro)':f1scm_mean, 'f1(weighted)':f1scw_mean, 'accuracy':acc_mean, 'AUC(ROC)':roc_mean, 'AUC(PR)':ap_mean}, ignore_index = True)


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 3))
cnf = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax1)
roc = RocCurveDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax2)
pr = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test).plot(ax=ax3)
plt.grid(False)
plt.tight_layout(pad=2.0)
# plt.show()
fig.savefig(f"{images_dir}/pot_mlp_{run}_{labeling}.png")