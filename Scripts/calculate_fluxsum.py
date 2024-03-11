fluxsum = pd.DataFrame()
for group_name, group_data in flux_yeast.groupby(['sample_id']):
# flux sum
  met_flux_sum = []
  v = np.array(group_data['pfba'])
  for ix, met_idx in enumerate(c_idx): #only for c compratment
    pr_rxns_idx = np.where(S[met_idx,:]>0)
    con_rxns_idx = np.where(S[met_idx,:]<0)
    pr_sum = 0
    con_sum = 0
    for j, rxn_idx in enumerate(pr_rxns_idx[0]):
      if v[rxn_idx] > 0:
        coef = S[ix,rxn_idx]
        pr_sum += coef * v[rxn_idx]
    for k, rxn_idx in enumerate(con_rxns_idx[0]):
      if v[rxn_idx] <0:
        coef = S[ix,rxn_idx]
        con_sum += coef * v[rxn_idx]
    met_sum = pr_sum + con_sum
    met_flux_sum.append(met_sum)

# save flux sums
  df_temp2 = pd.DataFrame(c_mets) # only for c compartment
  df_temp2 = df_temp2.rename(columns={0: 'metName'})
  df_temp2['sample_id'] = group_name
  df_temp2['flux_sum'] = met_flux_sum

  fluxsum = pd.concat([fluxsum, df_temp2], ignore_index=True)
