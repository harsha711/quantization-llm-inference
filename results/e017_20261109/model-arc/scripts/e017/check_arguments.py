"""Exercise the actual prerequisite parser before expensive sanitizer launches."""
import argparse
import ast
from pathlib import Path
import sys
path=Path(__file__).with_name('prerequisite.py')
block=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.If)][-1]
observed=[]
namespace=dict(__name__='__main__',argparse=argparse,Path=Path,REPRESENTATIVE='model.layers.0.mlp.down_proj',
    capture=lambda *x:None,prepare=lambda *x:None,check=lambda *x:observed.append(x))
for tool in ('memcheck','synccheck','racecheck'):
    sys.argv=['prerequisite.py','/tmp/results','--mode','check','--m','1',f'--suffix=-{tool}','--no-trace']
    exec(compile(ast.Module(body=[block],type_ignores=[]),str(path),'exec'),namespace)
    assert observed[-1][-2:]==('-'+tool,False)
print('PASS actual prerequisite parser accepts all three sanitizer suffixes')
