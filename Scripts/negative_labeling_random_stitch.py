labeling = 'rs'
stitch_ecoli = pd.read_csv('/content/gdrive/MyDrive/phd/PMI/review/stitch_cleaned.csv', header=0, index_col= 0)
# stitch_ecoli
ijo_met_cid = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/met_cid.xlsx', header=0)
# ijo_met_cid
ijo_met_cid = ijo_met_cid.dropna()
ijo_met_cid['CID'] = ijo_met_cid['CID'].astype(int)
ijo_met_cid = ijo_met_cid.drop_duplicates()
# ijo_met_cid
merge1 = data.merge(ijo_met_cid, how='inner',  left_on=['metabolite'], right_on = ['metName'])
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

met_stitch = merge3.loc[merge3['combined_score'] >= 500]['CID'].astype('int').tolist()
prot_stitch = merge3.loc[merge3['combined_score'] >= 500]['grRules'].tolist()
# met_stitch
# prot_stitch

merge3['class'] = np.nan

for index, row in merge3.iterrows():
  met = row['CID']
  # print(met)
  protein = row['grRules']
  # print(protein)
  if row['medium'] == 1:
    # print("1")
    merge3.at[index,'class'] = 1
  elif row['medium'] == 0:
    # print("0")
    if met in met_stitch:
      if protein in prot_stitch:
        merge3.at[index,'class'] = 0

print(merge3['class'].isna().sum())
print(merge3['class'].value_counts()[0], merge3['class'].value_counts()[1])

filtered_merge3 = merge3[merge3['class'].notnull()]
filtered_merge3 = filtered_merge3.loc[(filtered_merge3['flx_cv'] > 1) & (filtered_merge3['eta_cv'] > 1)]
filtered_merge3