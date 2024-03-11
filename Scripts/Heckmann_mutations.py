mutation = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/strain_mutation.xlsx',header=0, index_col=None)
# mutation
# mutation.columns
mutation_list = []
for column in mutation.columns:
  list1 = mutation[column].tolist()
  mutation_list.append(list1)
print(mutation_list)
mutation_bnumer = []
for idx, genes in enumerate(mutation_list):
  temp = []
  for gene in genes:
    if (iJO_bigg['name'] == gene).any():
      bnumber = iJO_bigg.loc[iJO_bigg['name'] == gene]['bigg_id'].values[0]
      temp.append(bnumber)
  mutation_bnumer.append(temp)
# mutation_bnumer

lb_strains = pd.DataFrame()
ub_strains = pd.DataFrame()

lb_df = pd.DataFrame(lb)
ub_df = pd.DataFrame(ub)

for i in range(0,18):
  lb_strains = pd.concat([lb_strains, lb_df], axis=1)
  ub_strains = pd.concat([ub_strains, ub_df], axis=1)

biomass_wt = 13
biomass_core = 18
for i in range(0,18):
  print("i: ",i)
  for index, row in gr_iJO.iterrows():
    reaction = row['reactions']
    rule = row['grRules']
    if rule in mutation_bnumer[i]:
      print("rule",rule)
      print("reaction",reaction)
      # print("lb",lb_pgi1[index])
      lb_strains.iloc[index,i] = 0
      # print("lb_pgi",lb_pgi1[index])
      # print("ub",ub_pgi1[index])
      ub_strains.iloc[index,i] = 0
      # print("ub_pgi",ub_pgi1[index])
  # set the growth rate for both wd and core biomass reaction, we will then remove one in the constraints
  lb_strains.iloc[13,i] = strain_gro.iloc[i,0]
  lb_strains.iloc[18,i] = strain_gro.iloc[i,0]
  ub_strains.iloc[13,i] = strain_gro.iloc[i,1]
  ub_strains.iloc[18,i] = strain_gro.iloc[i,1]
  # set golucose experimental value
  # lb_strains.iloc[EX_glc__D_e,i] = glc.iloc[i,0]
  ub_strains.iloc[EX_glc__D_e,i] = 1000
  # limit other carbon sources
  lb_strains.iloc[489,i] = 0
  lb_strains.iloc[237,i] = 0
  lb_strains.iloc[443,i] = 0
  lb_strains.iloc[546,i] = 0
  lb_strains.iloc[26,i] = 0
  lb_strains.iloc[136,i] = 0
  lb_strains.iloc[226,i] = 0
  lb_strains.iloc[426,i] = 0

  lb_strains.iloc[EX_ac_e,i] = 0
  lb_strains.iloc[EX_succ_e,i] = 0
  # # set acc experimental value
  # lb_strains.iloc[EX_ac_e,i] = ace_se.iloc[i,0]
  # ub_strains.iloc[EX_ac_e,i] = ace_se.iloc[i,1]
  # # set succ experimental value
  # lb_strains.iloc[EX_succ_e,i] = suc_se.iloc[i,0]
  # ub_strains.iloc[EX_succ_e,i] = suc_se.iloc[i,1]