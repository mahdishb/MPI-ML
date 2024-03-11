# Load iJO1366 model
mat = scipy.io.loadmat('/content/gdrive/MyDrive/phd/PMI/EtaFunction/model.mat')

S = mat['model']['S'][0][0]
print("S: ", S.shape)
# b = mat['iJO1366']['b'][0][0].reshape((1805,))
# print("b: ", b.shape)
# c = mat['iJO1366']['c'][0][0].reshape((2583,))
# print("c: ", c.shape)
# ub = mat['iJO1366']['ub'][0][0].reshape((2583,))
# print("ub: ", ub.shape)
# lb = mat['iJO1366']['lb'][0][0].reshape((2583,))
# print("lb: ", lb.shape)

def nparray_to_list(nparray):
    return [x[0][0] for x in nparray]

metNames_iJO = nparray_to_list(mat['model']['metNames'][0][0])
print("metNames_iJO: ", len(metNames_iJO))

mets_iJO = nparray_to_list(mat['model']['mets'][0][0])
print("mets_iJO: ", len(mets_iJO))

rxnNames_iJO = nparray_to_list(mat['model']['rxnNames'][0][0])
print("rxnNames_iJO: ", len(rxnNames_iJO))

rxns_iJO = nparray_to_list(mat['model']['rxns'][0][0])
print("rxns_iJO: ", len(rxns_iJO))

# # load iJO1366 grRules
# gr_iJO = pd.read_excel('/content/gdrive/MyDrive/phd/PMI/EtaFunction/grRules_iJO.xlsx',header=None)
# # gr_iJO
# # gr_iJO.columns
# gr_iJO = gr_iJO.rename(columns={0: 'grRules'})
# gr_iJO['reactions'] = rxns_iJO
# gr_iJO

# find metabolite idx for each compartments

c_idx = []
c_names = []
e_idx = []
e_names = []
p_idx = []
p_names = []
m_idx = []
m_names = []
n_idx = []
n_names = []
er_idx = []
er_names = []


for idx, s in enumerate(mets_iJO):
    if '[c]' in s:
      c_idx.append(idx)
      c_names.append(mets_iJO[idx])
    elif '[e]' in s:
      e_idx.append(idx)
      e_names.append(mets_iJO[idx])
    elif '[p]' in s:
      p_idx.append(idx)
      p_names.append(mets_iJO[idx])
    elif '[m]' in s:
      m_idx.append(idx)
      m_names.append(mets_iJO[idx])
    elif '[n]' in s:
      n_idx.append(idx)
      n_names.append(mets_iJO[idx])
    elif '[er]' in s:
      er_idx.append(idx)
      er_names.append(mets_iJO[idx])

print("number of mets: ", len(mets_iJO))
print("number of c: ", len(c_idx))
print("number of p: ", len(p_idx))
print("number of e: ", len(e_idx))
print("number of m: ", len(m_idx))
print("number of n: ", len(n_idx))
print("number of er: ", len(er_idx))


c_mets = [metNames_iJO[i] for i in c_idx]
p_mets = [metNames_iJO[i] for i in p_idx]
m_mets = [metNames_iJO[i] for i in m_idx]
n_mets = [metNames_iJO[i] for i in n_idx]
er_mets = [metNames_iJO[i] for i in er_idx]

print("common idx: ",len(set(c_mets) & set(p_mets))) # check if there is any common idx btw c and p

# cp_idx = c_idx + p_idx
# cp_names = c_names + p_names
# print("number of c+p: ", len(cp_idx))