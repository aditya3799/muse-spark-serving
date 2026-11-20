"""Read pinned AST directly; no SGLang installation/model import/compile required."""
import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[4]
MODEL=ROOT/'sglang/python/sglang/srt/models/dflash.py'
WORKER=ROOT/'sglang/python/sglang/srt/speculative/dflash_worker_v2.py'


def extract(path,names,cls=None,namespace=None):
    tree=ast.parse(path.read_text())
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    nodes=[]
    for name in names:
        n=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
        n.decorator_list=[]
        # annotations are evaluated lazily by the future import below
        nodes.append(n)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    env={} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return env


def selector_methods():
    env={'torch':torch,'nn':nn,'F':F}
    extract(MODEL,['_score_edges','_follow_maps'],namespace=env)
    extract(MODEL,['build_lattice','sample_path'],cls='CandidateSelector',namespace=env)
    return env
