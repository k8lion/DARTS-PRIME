import sys
import os
import genotypes
from genotypes import *
from graphviz import Digraph
from utils import COLORMAP


def plot(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, color="#%02x%02x%02x" % tuple([int(x * 256) for x in COLORMAP[op][0:3]]))

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)


if __name__ == '__main__':
  for folder in ["olympe", "osirim"]:
    for root, _, files in os.walk("/home/kaitlin/repos/admmdarts/"+folder):
      for file in files:
        path = os.path.join(root, file)
        if "/genotype.txt" in path:# and not os.path.isfile(os.path.join(os.path.split(path)[0],"normalgraph")):
          try:
            with open(path, "r") as f:
              geno_raw = f.read()
              genotype = eval(geno_raw)
            plot(genotype.normal, os.path.join(os.path.split(path)[0], "normalgraph"))
            plot(genotype.reduce, os.path.join(os.path.split(path)[0], "reducegraph"))
          except:
            print("failed", path)

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
