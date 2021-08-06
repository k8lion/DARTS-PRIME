import sys
import os
import genotypes
from genotypes import *
from graphviz import Digraph
from utils import COLORMAP
PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
def plot(filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='40', height='1.0', width='1.0', penwidth='4', fontname="times"),
      engine='dot'
      )
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')

  steps = 4

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for p in range(7):
    for i in range(steps):
      for j in range(0, i+2):
        if j == 0:
          u = "c_{k-2}"
        elif j == 1:
          u = "c_{k-1}"
        else:
          u = str(j-2)
        v = str(i)
        g.edge(u, v, color="#%02x%02x%02x" % tuple([int(x * 256) for x in COLORMAP[PRIMITIVES[p]][0:3]]))

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)


plot("/home/kaitlin/repos/admmdarts/src/basecellgraph")

'''
  genotype_name = "BATH"
  genotype_path = "osirim/exp/admmsched-6657587-20210308-205418/genotype.txt" #os.path.join(utils.get_dir(), args.genotype_path, 'genotype.txt')
  if os.path.isfile(genotype_path):
    with open(genotype_path, "r") as f:
      geno_raw = f.read()
      genotype = eval(geno_raw)
  else:
    try:
      genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
      print("{} is not specified in genotypes.py".format(genotype_name))


  plot(genotype.normal, os.path.join(os.path.split(genotype_path)[0],"normalgraph"))
  plot(genotype.reduce, os.path.join(os.path.split(genotype_path)[0],"reducegraph"))
'''
