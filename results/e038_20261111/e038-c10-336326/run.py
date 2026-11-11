#!/usr/bin/env python3
"""E023: replay the corrected-Marlin decode step as a CUDA graph (E015's per-token replay, Q modules).

run.py validate OUT GEN CONTEXT...                      one process, E018 dual modules, E021 gate on graph-Q
run.py time RESULTS OUT VARIANT CONTEXT SERIES ANCHOR   one cell; VARIANT in R, GR, Q, GQ (G = graph replay)
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402  (adds the E017 paths)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('e021_gate', Path(__file__).resolve().parent / 'validate.py')  # e014 also has a validate.py
_gate = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gate)
prefix, rel = _gate.prefix, _gate.rel  # E021 gate helpers
from infer import Runner, checked  # noqa: E402

I = D.I


class GraphRunner(Runner):
    """infer.Runner whose decode step can be one captured graph per key, replayed per token."""
    use_graph, key, replays, mask = False, 'x', 0, None

    def eager_decode(self, token, position, bias):
        if not self.use_graph:
            return super().eager_decode(token, position, bias)
        if self.mask is None:
            self.mask, self.graphs = torch.empty_strided(bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device), {}
        self.mask.copy_(bias)
        if self.key not in self.graphs:  # E015 procedure: warm on a side stream, capture once, same cache position
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    super().eager_decode(token, position, self.mask)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = super().eager_decode(token, position, self.mask)
            torch.cuda.synchronize()
            self.graphs[self.key] = (graph, out, [token.data_ptr(), position.data_ptr(), self.mask.data_ptr()])
        graph, out, pointers = self.graphs[self.key]
        assert pointers == [token.data_ptr(), position.data_ptr(), self.mask.data_ptr()]
        graph.replay()
        self.replays += 1
        return out

    def run(self, prompt, graph, key='x', **kw):
        self.use_graph, self.key, before = graph, key, self.replays
        row, logits = self.request(prompt, **kw)
        assert self.replays - before == (row['decode_forwards'] if graph else 0)
        return row, logits


def validate(out, gen, contexts):
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        q = I.Linear(name, r.bias, I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name]), 'Q', op).eval()
        parent, leaf = name.rsplit('.', 1)
        setattr(work.model.get_submodule(parent), leaf, D.Dual(r, q))
    verdict = dict(generated=gen, gpu=work.identity['gpu'], cases={})
    for context in contexts:
        runner = GraphRunner(work, 'R', context)
        anchor = {}
        for prompt in ('primary', 'heldout'):
            L, rows = {}, {}
            for label, mode, graph in (('F64', 'F64', False), ('Q', 'Q', False), ('GQ', 'Q', True), ('GQ2', 'Q', True),
                                       ('R', 'R', False), ('GR', 'R', True)):
                D.MODE = mode
                rows[label], logits = runner.run(prompt, graph, key=mode, generated=gen, capture_logits=True)
                L[label] = logits.numpy().copy()
            n, tokens_ok, flip = prefix(L['GQ'], L['F64'])
            try:
                checked(rows['GQ'], rows['GQ'], context, gen)  # structural state invariants
                state = True
            except AssertionError as error:
                state = repr(error)
            s = dict(compared_steps=n, flip=flip, gq_vs_f64_rel_l2=rel(L['GQ'], L['F64'], n), workspace=op.audit(),
                     reported=dict(gr_bitwise_r=bool(np.array_equal(L['GR'], L['R'])), q_vs_f64_rel_l2=rel(L['Q'], L['F64'], n),
                                   r_tokens_equal_f64=rows['R']['output_ids'] == rows['F64']['output_ids']))
            s['criteria'] = dict(finite=bool(np.isfinite(L['GQ']).all()), tokens=tokens_ok, rel_l2_vs_F64=s['gq_vs_f64_rel_l2'] <= .003,
                                 graph_bitwise_eager=bool(np.array_equal(L['GQ'], L['Q'])), repeat_bitwise=bool(np.array_equal(L['GQ'], L['GQ2'])),
                                 workspace=all(s['workspace'].values()), state=state is True)
            s['accepted'] = all(s['criteria'].values())
            verdict['cases'][f'{context}-{prompt}'] = s
            if prompt == 'primary':
                anchor = dict(R=L['R'], GR=L['GR'], Q=L['Q'], GQ=L['GQ'])
            print(context, prompt, 'ACCEPTED' if s['accepted'] else 'REJECTED', json.dumps(s), flush=True)
        np.savez_compressed(out / f'anchor-{context}.npz', **anchor)
    verdict['accepted'] = all(c['accepted'] for c in verdict['cases'].values())
    I.save(out / 'verdict.json', verdict)
    print('ALL ACCEPTED' if verdict['accepted'] else 'NOT ALL ACCEPTED', flush=True)
    assert verdict['accepted']


def timed(results, out, variant, context, series, anchor):
    base, graph = variant[-1], variant.startswith('G')
    work = I.Workload(results, base)
    runner = GraphRunner(work, base, context)
    reference = json.loads((results / f'validation_R_{context}_primary_0-output.json').read_text())
    row, logits = runner.run('primary', graph, capture_logits=True)
    checked(row, reference, context)
    # tie this separate process to the validated one
    assert np.array_equal(logits.numpy(), np.load(anchor / f'anchor-{context}.npz')[variant])
    rows = []
    for index in range(23):  # 3 warmups + 20 samples, E017 plan
        row, _ = runner.run('primary', graph)
        checked(row, reference, context)
        row.update(sample=index - 3, series=series, variant=variant, graph_replays=runner.replays)
        rows.append(row)
    work.weights_unchanged()
    tag = f'primary_{variant}_{context}_s{series}'
    with (out / (tag + '.jsonl')).open('x') as log:
        for row in rows:
            log.write(json.dumps(row) + '\n')
    kept = rows[3:]
    summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'ttft_s', 'e2e_s')}
    summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept),
                   peak_allocated=max(r['peak_allocated'] for r in kept), peak_reserved=max(r['peak_reserved'] for r in kept),
                   workspace=work.operator.audit() if work.operator else None, gpu=work.identity['gpu'])
    I.save(out / (tag + '.json'), summary)
    print('DONE', tag, json.dumps(summary), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        if sys.argv[1] == 'validate':
            target = Path(sys.argv[2])
            target.mkdir(parents=True, exist_ok=False)
            validate(target, int(sys.argv[3]), [int(c) for c in sys.argv[4:]])
        else:
            assert sys.argv[1] == 'time' and sys.argv[4] in ('R', 'GR', 'Q', 'GQ')
            timed(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), Path(sys.argv[7]))
