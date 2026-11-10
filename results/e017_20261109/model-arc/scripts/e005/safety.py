#!/usr/bin/env python3
"""Executable finite bounds/stripe model for E005's inspected specialization."""
import json
import math
from collections import defaultdict


def stripe_model(k,n,grid):
    kt,nt=k//128,n//128
    part2=nt
    part1=0
    if nt>grid:
        part2=nt%grid
        if part2*3<=grid: part2+=grid
        part1=(nt-part2)//grid
    iters=math.ceil(kt*part2/grid)
    uses=defaultdict(list)
    edges=[]
    maxoff=-1
    for b in range(grid):
        # Part1 tiles have no cross-block reduction or lock/scratch accesses.
        for j in range(part1):
            assert b+j*grid < nt-part2
        col=(iters*b)//kt
        row=(iters*b)%kt
        off=b if part2>=grid else (iters*b)//kt-1
        while col<part2:
            count=min(iters*(b+1)-(kt*col+row),kt-row)
            if count<=0: break
            first=iters*math.ceil(kt*col/iters)
            ns=1;idx=0
            if first<=kt*(col+1):
                delta=first-kt*col
                ns=math.ceil((kt-delta)/iters)+(delta>0)
                d=iters*b-first
                if d<0 or (delta==0 and d==0):idx=ns-1
                else:idx=ns-1-d//iters-(delta>0)
            if part2>=grid:
                if ns>1 and idx==ns-1:off+=1
            else:off+=1
            assert 0<=idx<ns
            if ns>1:
                assert 0<=off<grid
                uses[(off,col)].append((idx,b,row,count))
                maxoff=max(maxoff,off)
            col+=1;row=0
    for (off,col),v in uses.items():
        v.sort()
        assert [x[0] for x in v]==list(range(len(v)))
        assert sum(x[3] for x in v)==kt
        # Every acquire depends on a strictly larger block index. No cycle.
        for x,y in zip(v,v[1:]):
            assert x[1]>y[1]
            edges.append([y[1],x[1]])
    # c_size=16*128*4/16=512 int4 per lock. Even k fragments only.
    relative=[64*j+t for j in range(0,8,2) for t in range(64)]
    assert max(relative)==447
    high=(maxoff*512+447+1)*16 if maxoff>=0 else 0
    original_scratch=36*16*256*4
    assert high<=original_scratch
    return dict(K=k,N=n,grid=grid,part1_iterations=part1,part2_tiles=part2,
                stripe_iters=iters,reduction_slices=len(uses),max_lock=maxoff,
                scratch_end_bytes=high,scratch_capacity_bytes=original_scratch,
                wait_edges=edges)


def check():
    # All units here are int4 except where explicitly converted to bytes.
    a_writes=[(16*(t//16)+(t%16))^((t//16)%8) for t in range(128)]
    a_reads=[(2*i+16*((t%32)%8)+(t%32)//8+2*((t//32)//2)*2)^(((2*i+16*((t%32)%8)+(t%32)//8+2*((t//32)//2)*2)//16)%8) for t in range(256) if t%32<16 for i in range(2)]
    assert min(a_reads)>=0 and max(a_reads)<128
    assert sorted(a_writes)==list(range(128))
    red=[]
    for t in range(256):
        ri=t//64;rd=512*ri+t%64
        for j in range(0,8,2):
            if 2<=ri<4:red.append(64*j+rd-1024)
            if 1<=ri<2:red.extend([64*j+rd,64*j+rd-512])
            if ri==0:red.append(64*j+rd)
    assert max(red)==959 and min(red)>=0 and (max(red)+1)*16<=32768
    output_half=[]
    for t in range(64):
        base=136*((t%32)%4*2)+(t%32)//4+64*(t//32)
        for j in range(4):
            for offset in (0,8):output_half.extend([base+16*j+offset,base+16*j+offset+136])
    assert max(output_half)<2176
    layout=dict(B_overlay_bytes=[0,32768],scale_bytes=[32768,33792],A_bytes=[33792,41984],
                declared_reduction_bytes=4352,actual_block_reduction_span_bytes=15360,
                bias_enabled=False,zero_points=False,activation_scales=False,global_scale=False,
                stages=4,A_stage_bytes=2048,B_stage_bytes=8192,scale_stage_bytes=256,
                alignment=16,A_max_stage_int4=max(a_reads),block_reduce_max_int4=max(red),
                output_max_half=max(output_half))
    shapes=[(256,128),(256,256),(256,2048),(4096,4096),(4096,11008),(11008,4096)]
    return dict(layout=layout,stripes=[stripe_model(k,n,g) for k,n in shapes for g in (36,72)])


if __name__=='__main__': print(json.dumps(check(),indent=2))
