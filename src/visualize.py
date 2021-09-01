import os

from graphviz import Digraph

from utils import COLORMAP

# old version of visualize

ABBREV = {
    'max_pool_3x3': 'MAX3',
    'avg_pool_3x3': 'AVE3',
    'skip_connect': 'SKIP',
    'sep_conv_3x3': 'SEP3',
    'sep_conv_5x5': 'SEP5',
    'dil_conv_3x3': 'DIL3',
    'dil_conv_5x5': 'DIL5'
}
def plot_rnn(genotype, filename):
    return

def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='40', fontname="times", penwidth='4.5'),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='40', height='1.0', width='1.0',
                       penwidth='4.5', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=ABBREV[op], color="#%02x%02x%02x" % tuple([int(x * 256) for x in COLORMAP[op][0:3]]))

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", style="dashed")
    g.render(filename, view=False)


if __name__ == '__main__':
    for folder in ["olympe", "osirim", "d9wolympe"]:
        for root, _, files in os.walk("/home/kaitlin/repos/admmdarts/" + folder):
            for file in files:
                path = os.path.join(root, file)
                if "/genotype.txt" in path:  # and not os.path.isfile(os.path.join(os.path.split(path)[0],"normalgraph")):
                    try:
                        with open(path, "r") as f:
                            geno_raw = f.read()
                            genotype = eval(geno_raw)
                        plot(genotype.normal, os.path.join(os.path.split(path)[0], "normalgraphup"))
                        plot(genotype.reduce, os.path.join(os.path.split(path)[0], "reducegraphup"))
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
