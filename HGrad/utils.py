from graphviz import Digraph
import numpy as np

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_graph(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'}) #LR: left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        if n.label != '':
            label = f'{n.label}: '
        else:
            label = ''

        if n.shape == ():
            label += f'{n.data}'
        else:
            label += f'{n.shape}'


        dot.node(name = uid, label=label, shape='record')
        
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# utility function to compare manual to pytorch gradients
def cmp_torch(s, dt, t):
    ex = np.all(dt.grad == t.grad.numpy())
    app = np.allclose(dt.grad, t.grad.numpy())
    maxdiff = np.max(np.abs(dt.grad - t.grad.numpy()))
    #print(dt.grad)
    #print(t.grad.numpy())
    print(f'{s:15s} | exact {str(ex):5s} | app {str(app):5s} | maxdiff: {maxdiff}')
