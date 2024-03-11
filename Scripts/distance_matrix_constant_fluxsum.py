# Distance matrix with constant flux sum

# GRADIS
GRADIS_time_s = time.time()
features = pd.DataFrame()
k = 0
for group_name, group_data in merge_30.groupby(['reaction','metName']):
  start_time= time.time()
  row = []
  rxn = list(group_data['reaction'])[0]
  met = list(group_data['metName'])[0]
  row.append(rxn)
  row.append(met)
  # print(row)
  x = list(group_data['eta'])
  y = list(group_data['flux_sum'])
  cv = lambda x: np.std(x) / np.mean(x) * 100
  eta_cv = cv(group_data['eta'])
  flx_cv = cv(group_data['flux_sum'])
  row.append(eta_cv)
  row.append(flx_cv)
  eta_max = group_data['eta'].max()
  # flx_max = group_data['flux_sum'].max()
  # if flx_max == 0:
  #   flx_max = 1
  for i in range(0,len(x)):
    for j in range(i+1, len(x)):
      distance = pow((pow(((x[i]-x[j])/eta_max),2) + pow(1,2)),(0.5)) # Replace the distance of metabolite concentration with a constant value
      # print(distance)
      row.append(distance)
  # print(x)
  # print(y)
  features = features.append(pd.DataFrame([row]), ignore_index=True)
  end_time= time.time()
  k += 1
  if k % 10000 == 0:
    print(k, ':', end_time- GRADIS_time_s)
GRADIS_time_e = time.time()
GRADIS_time = GRADIS_time_e - GRADIS_time_s
features = features.rename(columns={0: 'reaction'})
features = features.rename(columns={1: 'metabolite'})
features = features.rename(columns={2: 'eta_cv'})
features = features.rename(columns={3: 'flx_cv'})