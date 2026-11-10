"""Inspect completed gates without accepting incomplete model validation."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts/e017').resolve()))
import analyze as a
r=Path(sys.argv[1]);a.prerequisite(r);a.source_manifest(r,'validation')
for label in ('native-contract-build','native-contract-check','artifact-preparation','argument-check'):
    a.success(r,label)
manifest=a.read(r/'artifacts/manifest.json');act=a.read(r/'one-layer.json')['activations']
assert len(manifest['layers'])==196
cases=[]
for name in ('model.layers.0.self_attn.q_proj','model.layers.0.self_attn.k_proj','model.layers.0.mlp.gate_proj',a.REPRESENTATIVE):
    for tool in (None,'memcheck','synccheck','racecheck'):
        for m in ((1,128) if tool=='racecheck' else (1,128,2048)):
            cases.append(a.operator_case(r,m,name,manifest['layers'][name],act,tool))
base=a.read(r/'validation_O_128.json')
for variant in ('O','R'):
    a.success(r,f'validation_{variant}_128');cell=a.read(r/f'validation_{variant}_128.json')
    a.identity_checked(cell['identity'],variant,base['identity']);a.compiler(cell);a.runner_storage(cell,128)
    assert [x['prompt'] for x in cell['records']]==['primary','heldout']*2
    assert len(a.lines(r/f'validation_{variant}_128.jsonl'))==4
    for i,item in enumerate(cell['records']):
        first=cell['records'][i%2];row=item['record']
        a.row_checked(row,first['record'],128,variant)
        a.accept_logits(item['repetition']);assert item['repetition']['bitwise_equal']
        assert row['cache_hashes']==first['record']['cache_hashes']
        assert a.read(r/f'validation_{variant}_128_{item["prompt"]}_{i}-weights.json')['unchanged']
q=a.read(r/'validation_Q_128_primary_0-diagnostics.json')
a.row_checked(q['record'],a.read(r/'validation_R_128_primary_0-output.json'),128,'Q')
assert q['errors']['bad_logits']==591 and q['errors']['finite']
assert a.read(r/'validation_Q_128_primary_0-weights.json')['unchanged']
assert not list(r.glob('primary_*_s*.jsonl')) and not list(r.glob('heldout_*_s*.jsonl'))
assert not list(r.glob('trace_Q_*')) and not list(r.glob('validation_*_2048*'))
assert not (r/'FULL_VALIDATION_INSPECTED').exists()
result=dict(status='completed gates inspected; full-model failure remains rejected',operator_cases=len(cases),ordinary_cases=sum(x['tool'] is None for x in cases),sanitizer_cases=sum(x['tool'] is not None for x in cases),complete_O_R_requests=8,failed_Q_requests=1,request_timing_samples=0)
Path('results/e017_20261109/completed-gates-inspection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
