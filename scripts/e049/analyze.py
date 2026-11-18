#!/usr/bin/env python3
"""E049 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

prereq: E045's prerequisite checks for four paths, metric/window checks and the first profile.
full: both timing series (descriptive table) and the eight cyclic profiles (counter table, verdicts by the
predeclared rules). Writes OUTDIR/summary.json and prints the analysis text."""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SHAPES = [(4096,11008),(11008,4096)]
MS = [4,16,64,128]
NAMES = ['marlin','tc','tc_m32','cublas_dq']
PROFILE_MS = [4,64]
STALLS = ['barrier','branch_resolving','dispatch_stall','drain','lg_throttle','long_scoreboard','math_pipe_throttle',
          'membar','mio_throttle','misc','no_instruction','not_selected','selected','short_scoreboard','sleeping',
          'tex_throttle','wait']
METRIC = dict(
    dur='gpu__time_duration.sum', dram_rd='dram__bytes_op_read.sum', dram_wr='dram__bytes_op_write.sum',
    dram_pct='dram__throughput.avg.pct_of_peak_sustained_elapsed', l2_bytes='lts__t_bytes.sum',
    l2_sm_bytes='l1tex__m_xbar2l1tex_read_bytes.sum', l2_tex_rd_sectors='lts__t_sectors_srcunit_tex_op_read.sum',
    l2_tex_hit='lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum', l2_tex_miss='lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum',
    l2_hit='lts__t_sector_hit_rate.pct', l2_rd_hit='lts__t_sector_op_read_hit_rate.pct',
    ld_req='l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum', ld_sec='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    cp_req='l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum',
    cp_sec='l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum',
    bytes_per_sector='smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
    ldg='smsp__sass_inst_executed_op_global_ld.sum', lds='smsp__sass_inst_executed_op_shared_ld.sum',
    sts='smsp__sass_inst_executed_op_shared_st.sum', ldgsts='smsp__inst_executed_op_ldgsts.sum',
    ldsm='smsp__inst_executed_op_ldsm.sum', inst='smsp__inst_executed.sum',
    hmma='sm__inst_executed_pipe_tensor_subpipe_hmma.sum',
    hmma_pct='sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active',
    issue_pct='smsp__issue_active.avg.pct_of_peak_sustained_active', occ='sm__warps_active.avg.pct_of_peak_sustained_active',
    occ_theo='sm__maximum_warps_per_active_cycle_pct', waves='launch__waves_per_multiprocessor',
    grid='launch__grid_size', block='launch__block_size', regs='launch__registers_per_thread',
    smem_dyn='launch__shared_mem_per_block_dynamic', smem_static='launch__shared_mem_per_block_static',
    ctas='sm__ctas_active.avg.per_cycle_elapsed', eligible='smsp__warps_eligible.avg.per_cycle_active',
    sm_active='sm__cycles_active.avg', sm_elapsed='sm__cycles_elapsed.avg',
    sm_pct='sm__throughput.avg.pct_of_peak_sustained_elapsed', l2_pct='lts__throughput.avg.pct_of_peak_sustained_elapsed',
    **{'st_'+s: f'smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio' for s in STALLS})


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def ok(d,label):
    assert (d/f'{label}.exit').read_text().strip() == '0', label


def check_validation(row,key='validation'):
    checks = row[key]
    assert len(checks) == 4, row['config']
    for c in checks:
        assert c['num_bad'] == 0 and c['checked'] == row['M']*row['N'], c
    assert row['cublas_math_mode'] == 16 and row['output_tail_untouched'] and row['workspace_guards']


def same_bits(rs):
    """tc_m32 must reproduce tc's output bits (E048) in every cell present in rs."""
    by = {(r['K'],r['N'],r['M'],r['config']): [c['output_sha256'] for c in r['validation']] for r in rs}
    n = 0
    for (k,nn,m,c),h in by.items():
        if c == 'tc_m32' and (k,nn,m,'tc') in by:
            assert h == by[(k,nn,m,'tc')], (k,nn,m); n += 1
    return n


def check_untimed(path,shapes,configs,ms=MS):
    rs = rows(path)
    assert sorted((r['K'],r['N'],r['M'],r['config']) for r in rs) == sorted(
        (k,n,m,c) for k,n in shapes for m in ms for c in configs), path
    for r in rs:
        check_validation(r); assert r['iters'] == 0
    same_bits(rs)
    return rs


def sass_blocks(path):
    blocks, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = re.sub(r'_GLOBAL__N__[0-9a-f]+_','_GLOBAL__N__',line.split()[-1]); blocks[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)',line):
            blocks[name].append(line.strip())
    return blocks


def raw(path):
    r = list(csv.reader(path.open()))
    return [dict(zip(r[0],x)) for x in r[2:] if x]


def num(x):
    try: return float(str(x).replace(',',''))
    except ValueError: return None


def calibration(d,tag):
    ok(d,f'calibration-{tag}')
    rs = raw(d/'ncu'/f'calibration-{tag}.raw.csv')
    assert len(rs) == 1 and 'add' in rs[0]['Kernel Name'].lower(), rs  # the unique final add_, not a fill
    return rs[0]['Kernel Name']


def profile(d,tag):
    """One cyclic profile: every replay pass validated, exactly one kernel of the expected path."""
    ok(d,f'profile_{tag}')
    impl,m,k,n = re.match(r'(tc|marlin)_M(\d+)_K(\d+)_N(\d+)$',tag).groups(); m,k,n = int(m),int(k),int(n)
    passes = rows(d/'ncu'/f'{tag}.jsonl')
    assert passes and all(p['post_profile_validation'] == p['validation'] == passes[0]['validation'] for p in passes)
    for p in passes:
        check_validation(p); check_validation(p,'post_profile_validation')
        assert (p['config'],p['M'],p['K'],p['N'],p['launch_skip'],p['launch_count']) == (impl,m,k,n,399,1)
        assert p['profile_cycles'] == 100 and p['profile_warmup'] == 10
    rs = raw(d/'ncu'/f'{tag}.raw.csv')
    assert len(rs) == 1, (tag,len(rs))
    name = rs[0]['Kernel Name']
    assert ('Marlin' in name) if impl == 'marlin' else 'w4a16_tc_kernel' in name, name
    c = {key: num(rs[0].get(col)) for key,col in METRIC.items()}
    missing = [key for key,v in c.items() if v is None]
    c.update(case=tag,impl=impl,M=m,K=k,N=n,kernel=name,passes=len(passes),missing=missing)
    c['dur_us'] = c['dur']/1e3
    c['unique_bytes'] = k*n//2+2*n*k//128+2*m*k+2*m*n
    c['dram_rd_over_unique'] = c['dram_rd']/c['unique_bytes']
    c['peak_gbps'] = (c['dram_rd']+c['dram_wr'])/c['dur']/(c['dram_pct']/100)  # bytes/ns = GB/s
    tot = sum(c['st_'+s] or 0 for s in STALLS) or 1
    c['stall_share'] = {s: 100*(c['st_'+s] or 0)/tot for s in STALLS}
    c['sectors_per_req'] = c['ld_sec']/c['ld_req'] if c['ld_req'] else None
    c['cp_sectors_per_req'] = c['cp_sec']/c['cp_req'] if c['cp_req'] else None
    c['l2_tex_hit_pct'] = 100*c['l2_tex_hit']/(c['l2_tex_hit']+c['l2_tex_miss']) if c['l2_tex_hit'] is not None else None
    c['sm_active_pct'] = 100*c['sm_active']/c['sm_elapsed']
    inst = d/'ncu'/f'{tag}.instances.csv'
    if inst.exists() and inst.stat().st_size:
        r = raw(inst)[0]
        cell = next((v for key,v in r.items() if key.startswith('sass__inst_executed_per_opcode')),None)
        c['opcodes'] = cell
    return c


def prereq(d):
    for label in ['project-build','project-ctest','project-gpu','e048-check','tc-build','agree-4096x11008',
                  'agree-11008x4096','smoke','validation','memcheck','synccheck','racecheck']:
        ok(d,label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    bench, lib = sass_blocks(d/'sass-tc-bench.txt'), sass_blocks(d/'sass-tc-lib.txt')
    assert len(lib) == 6 and lib.keys() <= bench.keys() and all(bench[k] == lib[k] and lib[k] for k in lib), sorted(lib)
    counts = {k: dict(HMMA=sum('HMMA' in x for x in v),LDG=sum('LDG' in x for x in v)) for k,v in lib.items()}
    for k,v in counts.items():  # E044: 32/32 per tc kernel; E048: 64/40 per tc_m32 kernel
        assert v == (dict(HMMA=64,LDG=40) if 'm32' in k else dict(HMMA=32,LDG=32)), counts
    res = {re.sub(r'_GLOBAL__N__[0-9a-f]+_','_GLOBAL__N__',m.group(1)): m.group(2) for m in re.finditer(
        r'Function (\S*w4a16_tc_(?:m32_)?kernel\S*):\s*\n\s*(REG:\d+ STACK:\d+ SHARED:\d+ LOCAL:\d+)',(d/'res-usage-lib.txt').read_text())}
    expect = {'ILb0ELb1E':64,'ILb0ELb0E':76,'ILb1ELb1E':68,'ILb1ELb0E':73}
    for k,v in res.items():
        if 'm32' in k:
            assert v == 'REG:114 STACK:0 SHARED:33792 LOCAL:0', (k,v)
        else:
            assert any(f'REG:{expect[t]} STACK:0 SHARED:17408 LOCAL:0' == v for t in expect if t in k), (k,v)
    assert len(res) == 6, res
    check_untimed(d/'smoke.jsonl',[(256,2048)],NAMES)
    val = check_untimed(d/'validation.jsonl',SHAPES,NAMES)
    bits = same_bits(val)
    assert bits == 8
    for r in val:
        name, m, ks = r['config'], r['M'], r['traced_kernels']
        assert len(ks) == 1 or name == 'cublas_dq', ks  # cuBLAS M4 4096x11008 adds splitKreduce (E011)
        if name == 'marlin': assert ks[0].startswith('void e005_marlin::Marlin<')
        elif name == 'tc' or (name == 'tc_m32' and m < 32): assert 'w4a16_tc_kernel<true' in ks[0], ks
        elif name == 'tc_m32': assert 'w4a16_tc_m32_kernel<false>' in ks[0], ks
    for tool in ['memcheck','synccheck']:
        check_untimed(d/f'{tool}.jsonl',SHAPES,NAMES)
        assert 'ERROR SUMMARY: 0 errors' in (d/f'{tool}.log').read_text(), tool
    check_untimed(d/'racecheck.jsonl',SHAPES,['tc','tc_m32'])
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (d/'racecheck.log').read_text()
    agree = list(csv.DictReader((d/'agree.csv').open()))
    assert len(agree) == 4 and all(a['pass'] == '1' and a['M'] == '128' and a['seed'] == '3' for a in agree)
    out = []
    for a in agree:
        k,n = int(a['K']),int(a['N'])
        v = next(r for r in val if (r['K'],r['N'],r['M'],r['config']) == (k,n,128,a['kernel']))['validation'][0]
        assert float(a['max_abs_err']) == float(f"{v['max_abs']:.4e}") and float(a['rel_l2_err']) == float(f"{v['rel_l2']:.4e}"), (a,v)
        out.append(dict(kernel=a['kernel'],K=k,N=n,max_abs=v['max_abs'],rel_l2=v['rel_l2']))
    assert (d/'metrics-dropped.txt').read_text() == ''
    cal = {t: calibration(d,t) for t in ('e011','e049')}
    first = profile(d,'tc_M4_K4096_N11008')
    print('PREREQ PASS',json.dumps(dict(sass_counts=counts,resources=res,agreement=out,tc_m32_bitwise_cells=bits,
        calibration=cal,first_profile={k: first[k] for k in ('kernel','passes','dur_us','dram_rd','missing','opcodes') if k in first})))


def order(k,n,m,series):  # mirror of bench.py
    shift = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def cells(series_rows):
    out = {}
    for s,rs in series_rows.items():
        assert len(rs) == 32
        for r in rs:
            assert r['series'] == s and r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
            assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
            assert r['order'] == order(r['K'],r['N'],r['M'],s).index(r['config'])
            check_validation(r); check_validation(r,'post_timing_validation')
            out[(r['K'],r['N'],r['M'],r['config'],s)] = r['median_us']/4
        assert same_bits(rs) == 8
    return out


def timing_table(med):
    table = []
    for k,n in SHAPES:
        for m in MS:
            row = dict(K=k,N=n,M=m)
            for s in (1,2):
                mr = med[(k,n,m,'marlin',s)]
                row[f's{s}'] = {c+'_us': med[(k,n,m,c,s)] for c in NAMES}
                row[f's{s}'].update({f'{c}_fraction_of_marlin_speed': mr/med[(k,n,m,c,s)] for c in NAMES if c != 'marlin'})
                row[f's{s}']['tc_m32_over_tc'] = med[(k,n,m,'tc_m32',s)]/med[(k,n,m,'tc',s)]
            table.append(row)
    return table


def verdicts(p):
    out = {}
    for k,n in SHAPES:
        t4,t,mr,m4 = (p[f'{i}_M{m}_K{k}_N{n}'] for i,m in (('tc',4),('tc',64),('marlin',64),('marlin',4)))
        gap = t['dur_us']-mr['dur_us']
        extra_us = (t['dram_rd']-mr['dram_rd'])/(t4['peak_gbps']*1e3)  # bytes / (bytes per us)
        if t['dram_rd_over_unique'] < 1.2: a = 'REJECTED'
        elif t['dram_rd_over_unique'] >= 1.5 and mr['dram_rd_over_unique'] <= 1.15:
            a = 'CONFIRMED' if extra_us >= 0.5*gap else 'PARTIAL'
        else: a = 'INCONCLUSIVE'
        ls_t, ls_m = t['stall_share']['long_scoreboard'], mr['stall_share']['long_scoreboard']
        if ls_t < 15 or ls_t <= ls_m: b = 'REJECTED'
        elif (mr['ldgsts'] or 0) > 0 and ls_t >= 30 and ls_t >= 2*ls_m: b = 'CONFIRMED'
        else: b = 'PARTIAL'
        ir = t['inst']/mr['inst']
        c = 'REJECTED' if ir <= 1.2 else 'CONFIRMED' if ir >= 1.5 and t['issue_pct'] >= 40 else 'PARTIAL'
        out[f'K{k}xN{n}'] = dict(
            a=dict(verdict=a,tc_dram_over_unique=t['dram_rd_over_unique'],marlin_dram_over_unique=mr['dram_rd_over_unique'],
                   extra_dram_us=extra_us,profiled_gap_us=gap,share=extra_us/gap if gap else None,peak_gbps=t4['peak_gbps']),
            b=dict(verdict=b,tc_long_sb_share=ls_t,marlin_long_sb_share=ls_m,marlin_ldgsts=mr['ldgsts']),
            c=dict(verdict=c,inst_ratio=ir,tc_issue_pct=t['issue_pct'],tc_occ=t['occ'],marlin_occ=mr['occ'],
                   tc_waves=t['waves'],marlin_waves=mr['waves']),
            m4=dict(gap_us=t4['dur_us']-m4['dur_us'],tc_dram_pct=t4['dram_pct'],marlin_dram_pct=m4['dram_pct'],
                    tc_dram_over_unique=t4['dram_rd_over_unique'],marlin_dram_over_unique=m4['dram_rd_over_unique']))
    return out


def text(summary):
    L = ['E049 per-GEMM cycle medians (us; four-matrix cycle median / 4), series 1 | series 2; fraction = Marlin/path',
         'K     N     M    tc            tc_m32        Marlin        cuBLAS        frac tc       frac tc_m32   frac cuBLAS   m32/tc']
    for r in summary['timing']:
        f = lambda key,fmt: ' | '.join(fmt.format(r[s][key]) for s in ('s1','s2'))
        L.append(f"{r['K']:<5} {r['N']:<5} {r['M']:<4} {f('tc_us','{:6.1f}')}  {f('tc_m32_us','{:6.1f}')}  {f('marlin_us','{:6.1f}')}  "
                 f"{f('cublas_dq_us','{:6.1f}')}  {f('tc_fraction_of_marlin_speed','{:5.3f}')}  "
                 f"{f('tc_m32_fraction_of_marlin_speed','{:5.3f}')}  {f('cublas_dq_fraction_of_marlin_speed','{:5.3f}')}  "
                 f"{f('tc_m32_over_tc','{:5.3f}')}")
    L.append('')
    L.append('case|kernel|passes|replay us|DRAM rd MB (x unique)|DRAM wr MB|DRAM %|L2->SM MB|L2 MB|L2 tex-read hit %|occ ach/theo %|'
             'waves|grid|regs|smem dyn/static|CTAs/SM|SM active %|issue %|HMMA pipe %|HMMA|warp inst M|LDG|LDGSTS|LDSM|LDS|STS|'
             'sect/req ld (cp)|top stalls %')
    for c in summary['profiles'].values():
        g = lambda v,p=1: 'n/a' if v is None else f'{v:.{p}f}'
        top = ', '.join(f'{s} {v:.0f}' for s,v in sorted(c['stall_share'].items(),key=lambda x:-x[1])[:4])
        L.append('|'.join([c['case'],c['kernel'][:28],str(c['passes']),g(c['dur_us']),
            f"{g(c['dram_rd']/1e6,2)} ({g(c['dram_rd_over_unique'],2)})",g(c['dram_wr']/1e6,2),g(c['dram_pct']),
            g(c['l2_sm_bytes']/1e6),g(c['l2_bytes']/1e6),g(c['l2_tex_hit_pct']),f"{g(c['occ'])}/{g(c['occ_theo'])}",
            g(c['waves'],2),g(c['grid'],0),g(c['regs'],0),f"{g(c['smem_dyn'],0)}/{g(c['smem_static'],0)}",g(c['ctas'],2),
            g(c['sm_active_pct']),g(c['issue_pct']),g(c['hmma_pct']),g(c['hmma'],0),g(c['inst']/1e6,3),g(c['ldg'],0),
            g(c['ldgsts'],0),g(c['ldsm'],0),g(c['lds'],0),g(c['sts'],0),
            f"{g(c['sectors_per_req'],2)} ({g(c['cp_sectors_per_req'],2)})",top]))
        if c['missing']: L.append(f"   missing metrics: {c['missing']}")
        if c.get('opcodes'): L.append(f"   opcodes: {c['opcodes'][:600]}")
    L.append('')
    for shape,v in summary['verdicts'].items():
        L.append(f'{shape}: ' + json.dumps(v))
    return '\n'.join(L)


def full(d,out):
    med = cells({s: rows(d/f'series{s}.jsonl') for s in (1,2)})
    profiles = {f'{i}_M{m}_K{k}_N{n}': profile(d,f'{i}_M{m}_K{k}_N{n}') for k,n in SHAPES for m in PROFILE_MS for i in ('tc','marlin')}
    summary = dict(timing=timing_table(med),profiles=profiles,verdicts=verdicts(profiles))
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(text(summary))


def self_test():
    for k,n in SHAPES:
        for m in MS:
            assert order(k,n,m,2) == order(k,n,m,1)[::-1] and sorted(order(k,n,m,1)) == sorted(NAMES)
    offsets = [((SHAPES.index(s)+MS.index(m)) % 4) for s in SHAPES for m in MS]
    assert sorted(offsets) == [0,0,1,1,2,2,3,3]
    base = dict(dur_us=100,dram_rd=23e6,dram_wr=0,dram_pct=80,inst=1e6,issue_pct=30,occ=40,waves=1,ldgsts=1,
                stall_share={s: 0 for s in STALLS},dram_rd_over_unique=1.0,peak_gbps=440)
    def case(**kw): c = json.loads(json.dumps(base)); c.update(kw); return c
    p = {}
    for k,n in SHAPES:
        p[f'tc_M4_K{k}_N{n}'] = case()
        p[f'marlin_M4_K{k}_N{n}'] = case()
        p[f'marlin_M64_K{k}_N{n}'] = case(stall_share={**base['stall_share'],'long_scoreboard':5})
        p[f'tc_M64_K{k}_N{n}'] = case(dur_us=140,dram_rd=69e6,dram_rd_over_unique=3.0,inst=2e6,issue_pct=45,
                                      stall_share={**base['stall_share'],'long_scoreboard':40})
    v = verdicts(p)['K4096xN11008']
    assert (v['a']['verdict'],v['b']['verdict'],v['c']['verdict']) == ('CONFIRMED','CONFIRMED','CONFIRMED'), v
    p['tc_M64_K4096_N11008'].update(dram_rd_over_unique=1.1,inst=1.1e6,stall_share={**base['stall_share'],'long_scoreboard':4})
    v = verdicts(p)['K4096xN11008']
    assert (v['a']['verdict'],v['b']['verdict'],v['c']['verdict']) == ('REJECTED','REJECTED','REJECTED'), v
    print('SELF-TEST PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir',type=Path,nargs='?')
    ap.add_argument('--stage',choices=['prereq','full'])
    ap.add_argument('--out',type=Path)
    ap.add_argument('--self-test',action='store_true')
    a = ap.parse_args()
    if a.self_test: self_test()
    elif a.stage == 'prereq': prereq(a.dir)
    else: full(a.dir,a.out or a.dir)
