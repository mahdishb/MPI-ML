# parameters = {
#               'max_depth': 20,
#               'max_features': 'sqrt',
#               'min_samples_leaf': 1,
#               'min_samples_split': 2,
#               'n_estimators': 600}

clf_rf = RandomForestClassifier()

rus= RandomUnderSampler()
X_train_res, y_train_res= rus.fit_resample(x_train, y_train)

clf_rf.fit(X_train_res, y_train_res)
# clf_rf.fit(x_train, y_train)

prediction = clf_rf.predict(x_test)

acc = accuracy_score(y_test, prediction)
f1sc = f1_score(y_test, prediction)
prec = precision_score(y_test, prediction)
reca = recall_score(y_test, prediction)
# roc = roc_auc_score(np.ravel(y_test), prediction)

print("accuracy & f1score $ precision & recall:", (acc,f1sc,prec,reca))
# genes = X_train_res.columns
feature_importances = pd.DataFrame(clf_rf.feature_importances_, index =X_train_res.columns,  columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head()
feature_names = X_train_res.columns
importances = clf_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
fig.savefig(f"{images_dir}/FI_{run}_{labeling}.png")