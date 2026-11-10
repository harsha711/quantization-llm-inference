import numpy as np
z = np.load('validate-333424/out-128/logits-heldout.npz')
print(z.files)
L = {k: z[k].astype(np.float64).reshape(len(z[k]), -1) for k in z.files}
T = {k: v.argmax(1) for k, v in L.items()}
for k in T: print(k, 'equal R:', (T[k] == T['R']).all(), 'equal F64:', (T[k] == T['F64']).all())
i = int(np.argmax(T['Q'] != T['R'])); print('first divergence index', i)
for k in ('R', 'Q', 'F32', 'F64'):
    s = np.sort(L[k][i])[::-1]; print(k, 'top1', T[k][i], 'margin top1-top2', s[0] - s[1], 'logit of R-token', L[k][i][T['R'][i]], 'logit of Q-token', L[k][i][T['Q'][i]])
d = np.linalg.norm(L['Q'][:i+1] - L['R'][:i+1]) / np.linalg.norm(L['R'][:i+1]); print('Q vs R rel L2 up to divergence', d)
