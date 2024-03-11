smiles_yeast = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/met_smiles_yeast.xlsx', header=0)
# smiles_yeast

stitch_ecoli = pd.read_csv('/content/gdrive/MyDrive/phd/PMI/review/yeast_stitch_cleaned.csv', header=0, index_col= 0)
# stitch_ecoli
ijo_met_cid = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/met_cid_yeast.xlsx', header=0)
# ijo_met_cid
ijo_met_cid = ijo_met_cid.dropna()
ijo_met_cid['CID'] = ijo_met_cid['CID'].astype(int)
ijo_met_cid = ijo_met_cid.drop_duplicates()
# ijo_met_cid
gr_iJO = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/gr_yeast.xlsx', header=0)
met_map = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/metNames.xlsx', header=0)
merge0 = data.merge(met_map, how='inner',  left_on=['metabolite'], right_on = ['model'])
# merge0
# ijo_met_cid
merge1 = ijo_met_cid.merge(merge0, how='inner',  left_on=['metName'], right_on = ['new'])
# merge1
merge2 = merge1.merge(gr_iJO, how='inner',  left_on=['reaction'], right_on = ['reactions'])
# merge2
# merge3 = merge2.merge(stitch_ecoli,  how='inner', left_on=['grRules','CID'], right_on = ['protein','chemical'])
# merge3
# merge3['combined_score'].isna().sum()
merge3 = merge2.merge(stitch_ecoli,  how='left', left_on=['grRules','CID'], right_on = ['protein','chemical'])
# merge3

print(merge3['combined_score'].isna().sum())
merge3['combined_score'] = merge3['combined_score'].fillna(0)
# merge3
merge3.info()
merge3['medium'] = 0
merge3['high'] = 0
# merge3
merge3.loc[merge3['combined_score'] >= 500, 'medium'] = 1
merge3.loc[merge3['combined_score'] < 500, 'medium'] = 0
# merge3
merge3.loc[merge3['combined_score'] >= 700, 'high'] = 1
merge3.loc[merge3['combined_score'] < 700, 'high'] = 0
merge3 = merge3.drop_duplicates()
# merge3
print(merge3['medium'].value_counts()[1], merge3['high'].value_counts()[1])
print(merge3['medium'].value_counts()[0], merge3['high'].value_counts()[0])
merge3 = merge3.drop(columns=['model', 'new'])
# merge3
merge4 = merge3.merge(smiles_yeast, how='inner',  left_on=['CID'], right_on = ['CID'])
# merge4
filtered_merge4 = merge4.loc[(merge4['flx_cv'] > 1) & (merge4['eta_cv'] > 1)]
filtered_merge4

positives = merge4.loc[merge4['combined_score'] >= 500].copy()
met_stitch = positives['CID'].astype('int').tolist()
prot_stitch = positives['grRules'].tolist()

def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    # fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=1024)
    fp1 = AllChem.RDKFingerprint(mol1, fpSize = 64)
    # fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=1024)
    fp2 = AllChem.RDKFingerprint(mol2, fpSize = 64)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

filtered_merge4['class'] = np.nan
zero_t = 0
GRADIS_time_s = time.time()
k = 0
for index, row in filtered_merge4.iterrows():
  met = row['CID']
  protein = row['grRules']
  if row['medium'] == 1:
    filtered_merge4.at[index,'class'] = 1
  elif row['medium'] == 0:
    if protein in prot_stitch:
      mpi = positives.loc[positives['protein'] == protein]
      for index_x, row_x in mpi.iterrows():
        smi1 = row['smiles']
        smi2 = row_x['smiles']
        tanimoto_score = tanimoto_calc(smi1, smi2)
        if tanimoto_score == 0:
          filtered_merge4.at[index,'class'] = 0
          zero_t = zero_t + 1
          break
  end_time = time.time()
  k = k + 1
  if index % 1000 == 0:
    print(index, ':', end_time- GRADIS_time_s)


# print(filtered_merge4['class'].isna().sum())
# print(filtered_merge4['class'].value_counts()[0], filtered_merge4['class'].value_counts()[1])

print(filtered_merge4['class'].isna().sum())
print(filtered_merge4['class'].value_counts()[0])
print(filtered_merge4['class'].value_counts()[1])