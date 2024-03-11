labeling = 'potential'
num_pos = merge3['medium'].value_counts()[1]
num_neg = merge3['medium'].value_counts()[0]
num_tot = len(merge3)
itr = num_neg / num_pos
print(num_pos, num_neg, itr, num_tot)

idx_pos = merge3.loc[merge3["medium"] == 1]
idx_neg = merge3.loc[merge3["medium"] == 0]
neg_data = idx_neg.iloc[:,s_idx:e_idx].copy()
neg_score = idx_neg[['reaction','metabolite']].copy()

for i in range(0,20):
  print("i: ", i)
  x_pos = idx_pos.iloc[:,s_idx:e_idx].copy()
  y_pos = idx_pos.iloc[:,y_idx].copy()
  x_neg = idx_neg.iloc[:,s_idx:e_idx].copy()
  y_neg = idx_neg.iloc[:,y_idx].copy()

  x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(x_pos, y_pos, train_size=800, shuffle=True) #random_state=42
  x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(x_neg, y_neg, train_size=800, shuffle=True) #random_state=42

  x_data = pd.concat([x_train_pos, x_train_neg], ignore_index=True)
  y_data = pd.concat([y_train_pos, y_train_neg], ignore_index=True)

  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True) #random_state=42

  parameters = {'C': 0.1, 'gamma': 10, 'kernel': 'rbf', 'cache_size': 7000}
  # clf_svm = SVC(kernel='linear')
  clf_svm = SVC(**parameters)
  clf_svm.fit(x_train, y_train)

  prediction = clf_svm.predict(x_test)

  acc = accuracy_score(y_test, prediction)
  f1sc = f1_score(y_test, prediction)
  prec = precision_score(y_test, prediction)
  reca = recall_score(y_test, prediction)

  # print(classification_report(y_test, prediction))
  print("accuracy & f1score $ precision & recall:", (acc,f1sc,prec,reca))

  scores = clf_svm.predict(neg_data)

  neg_score['itr '+str(i)] = scores

neg_score['score'] = neg_score.sum(axis=1)
potential_neg = neg_score.loc[neg_score["score"] == 0]
train_idx = [*potential_neg.index.values, *idx_pos.index.values]

x_data_potneg = x_data.loc[train_idx]
y_data_potneg = y_data.loc[train_idx]
x_data_potneg.to_csv(f'/content/gdrive/MyDrive/phd/PMI/EtaFunction/x_data_potneg_{run}.csv')
y_data_potneg.to_csv(f'/content/gdrive/MyDrive/phd/PMI/EtaFunction/y_data_potneg_{run}.csv')