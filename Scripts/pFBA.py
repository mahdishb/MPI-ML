eta_data = pd.DataFrame()
fluxsum = pd.DataFrame()
for i in range(0,18):

  print("i: ", i)
  removed_biomass = 13 # 13 is wt and 18 is core
  lowerb = np.array(lb_strains.iloc[:,i]).copy()
  upperb = np.array(ub_strains.iloc[:,i]).copy()
  lowerb[removed_biomass] = 0
  upperb[removed_biomass] = 0

# Define the enviroment
  env = gp.Env(empty=True, params=params)
  env.setParam('OutputFlag', 0)
  env.start()

# Create model
  model_pfba = gp.Model('pFBA', env=env)

# Decision variables
  v = model_pfba.addMVar(c.shape, vtype=GRB.CONTINUOUS, lb=lowerb, ub=upperb, name="fluxes")
  t = model_pfba.addMVar(c.shape, vtype=GRB.CONTINUOUS, name="norm")

# Constraints
  const1_steady_state = model_pfba.addMConstr(S, v, '=', b, name="steady_state")
  # const2_biomass2 = model_pfba.addConstr(v[removed_biomass] == 0, name="Biomass_wt")
  const3_temp_norm = model_pfba.addConstr(v <= t, name="norm_side1")
  const4_temp_norm = model_pfba.addConstr(-v <= t, name="norm_side2")

# Objective function
  model_pfba.setObjective(t.sum(), GRB.MINIMIZE)

# Run optimization engine
# model_fba.params.OutputFlag = 0
  model_pfba.optimize()

# save the results
  if model_pfba.Status == 3:
    print("Status", model_pfba.Status)
  else:
    print("obej", model_pfba.ObjVal)
    print("non zero v", np.count_nonzero(v.x))

# save v s
  df_temp = pd.DataFrame(v.x)
  df_temp = df_temp.rename(columns={0: 'pfba'})
  df_temp['sample_id'] = strain_name[i]
  df_temp['reaction'] = rxns_iJO

  eta_data = pd.concat([eta_data, df_temp], ignore_index=True)

# flux sum
  met_flux_sum = []
  for ix, met_idx in enumerate(c_idx): #only for c compratment
    pr_rxns_idx = np.where(S[met_idx,:]>0)
    con_rxns_idx = np.where(S[met_idx,:]<0)
    pr_sum = 0
    con_sum = 0
    for j, rxn_idx in enumerate(pr_rxns_idx[0]):
      if v.x[rxn_idx] > 0:
        coef = S[ix,rxn_idx]
        pr_sum += coef * v.x[rxn_idx]
    for k, rxn_idx in enumerate(con_rxns_idx[0]):
      if v.x[rxn_idx] <0:
        coef = S[ix,rxn_idx]
        con_sum += coef * v.x[rxn_idx]
    met_sum = pr_sum + con_sum
    met_flux_sum.append(met_sum)

# save flux sums
  df_temp2 = pd.DataFrame(c_mets) # only for c compartment
  df_temp2 = df_temp2.rename(columns={0: 'metName'})
  df_temp2['sample_id'] = strain_name[i]
  df_temp2['flux_sum'] = met_flux_sum

  fluxsum = pd.concat([fluxsum, df_temp2], ignore_index=True)


  model_pfba.reset(0)
