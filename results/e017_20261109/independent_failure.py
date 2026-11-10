import json,hashlib,sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1])
load=lambda name: np.load(r/name,allow_pickle=False)['logits']
q=load('validation_Q_128_primary_0-output.npz');ref=load('validation_R_128_primary_0-output.npz')
assert q.shape==ref.shape==(64,1,151936)
delta=np.abs(q.astype(np.float32)-ref.astype(np.float32))
bad=delta>np.float32(.03)+np.float32(.01)*np.abs(ref.astype(np.float32))
idx=np.argwhere(bad)
assert np.array_equal(idx,np.load(r/'validation_Q_128_primary_0-reference-failed-indices.npz')['indices'])
stats=json.loads((r/'validation_Q_128_primary_0-reference-errors.json').read_text())
assert len(idx)==stats['bad_logits']==591
assert hashlib.sha256(q.tobytes()).hexdigest()==stats['candidate_sha256']
assert hashlib.sha256(ref.tobytes()).hexdigest()==stats['reference_sha256']
rows={v:json.loads((r/f'validation_{v}_128_primary_0-output.json').read_text()) for v in ('O','R','Q')}
assert rows['Q']['output_ids']==rows['R']['output_ids']
assert rows['Q']['positions_seen']==list(range(128,191))
assert rows['Q']['feedback']==rows['Q']['output_ids'][:-1]
assert rows['Q']['cache_lengths']==[191]*28
assert all(rows['Q']['cache_reset_zero']) and rows['Q']['storage_stable'] and rows['Q']['cache_finite'] and rows['Q']['cache_tail_zero']
assert all(rows['Q']['operator_workspace'].values())
assert json.loads((r/'validation_Q_128_primary_0-weights.json').read_text())['unchanged']
setups={v:json.loads((r/f'validation_{v}_128-setup.json').read_text()) for v in ('O','R','Q')}
resident={v:setups[v]['identity']['resident']['bytes'] for v in setups}
quant=[]
for prompt,index in [('primary',0),('heldout',1)]:
 o=json.loads((r/f'validation_O_128_{prompt}_{index}-output.json').read_text())
 x=json.loads((r/f'validation_R_128_{prompt}_{index}-output.json').read_text())
 quant.append(dict(prompt=prompt,unequal_token_positions=[i for i,(a,b) in enumerate(zip(o['output_ids'],x['output_ids'])) if a!=b],errors=json.loads((r/f'validation_R_128_{prompt}_{index}-quantization.json').read_text())))
result=dict(status='independently reproduced failed elementwise gate; no timing accepted',shape=list(q.shape),bad_logits=int(bad.sum()),total_logits=q.size,finite=bool(np.isfinite(q).all() and np.isfinite(ref).all()),max_abs=float(delta.max()),relative_l2=float(np.linalg.norm(delta.astype(np.float64))/np.linalg.norm(ref.astype(np.float64))),bad_by_generated_position=bad.reshape(64,-1).sum(axis=1).tolist(),greedy_tokens_equal=True,cache_and_feedback_checks=True,weights_unchanged=True,model_operator_resident_bytes=resident,runner_resident_bytes={v:setups[v]['runner_resident']['bytes'] for v in setups},resident_reduction_pct=100*(1-resident['Q']/resident['R']),request_allocator_bytes={v:{k:rows[v][k] for k in ('allocated','reserved','peak_allocated','peak_reserved')} for v in rows},setup_costs={v:{k:setups[v]['identity'].get(k) for k in ('load_s','replace_s','load_peak_allocated','replace_peak_allocated')} for v in setups},quantization_changes=quant,limits=['First Q128 primary request only; no repeated Q or held-out Q request completed','No context2048 model validation or complete-model coverage trace','Resident storage is observed setup storage, not accepted warmed inference memory','O/R later free-running differences include changed conditioning'])
Path('results/e017_20261109/independent-model-failure.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('quantization_changes',)},indent=2))
