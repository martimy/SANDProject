# Copyright (c) 2024 Maen Artimy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io


from rich.console import Console
from rich.table import Table


def design_output(out, cost, labels, cap):
    backbone = out["backbone"]
    mesh = out["mesh"]
    chlist = out["channels"]

    total = 0

    # Create a console object
    console = Console()

    table = Table(title="Network Links")

    table.add_column("From", min_width=5)
    table.add_column("To", min_width=5)
    table.add_column("Capacity", min_width=10)
    table.add_column("Cost", min_width=10)

    for i in range(len(mesh)):
        x, y = mesh[i]
        acost = cost[x][y] * chlist[i]
        total += acost
        table.add_row(labels[x], labels[y], str(chlist[i] * cap), str(acost))

    console = Console()
    console.print(table)

    # Print total cost
    console.print(f"{'Total cost = ':<12}{total:<8}")
    console.print(f"Number of backbone nodes = {len(backbone)}")

    # Calculate and print number of links in the backbone
    bknet = [p for p in mesh if p[0] in backbone and p[1] in backbone]
    console.print(f"Number of links in the backbone = {len(bknet)}")


def cost_to_dataframe(out, cost, labels):
    backbone = out["backbone"]
    mesh = out["mesh"]
    chlist = out["channels"]

    data = []

    for i in range(len(mesh)):
        x, y = mesh[i]
        cost_val = cost[x][y] * chlist[i]
        data.append([labels[x], labels[y], chlist[i], cost_val])

    total_cost = sum(row[3] for row in data)

    bknet = [(x, y) for (x, y) in mesh if x in backbone and y in backbone]

    return (
        pd.DataFrame(data, columns=["From", "To", "Ch", "Cost"]),
        total_cost,
        len(backbone),
        len(bknet),
    )


def plot_network(out, numNodes, labels=[], pos=None, edisp=True, image=None):
    mesh = out["mesh"]
    ch = out["channels"]
    backbone = out["backbone"]
    median = out["median"]
    tree = out["tree"]

    # Separate the mesh
    bknet = [p for p in mesh if p[0] in backbone and p[1] in backbone]
    local = [p for p in mesh if p not in bknet]

    # plt.figure(figsize=(6, 6), facecolor="white")

    fig, ax = plt.subplots()  # figsize=(6, 6))

    if image is not None:
        img = Image.open(io.BytesIO(image))
        img_array = np.array(img)
        ax.imshow(img_array, cmap="gray", extent=[-1, 1, -1, 1], aspect="auto")

    G = nx.path_graph(numNodes)
    G.add_edges_from(local)
    G.add_edges_from(bknet)

    if pos is None:
        pos = nx.spring_layout(G)

    # nx.draw_networkx_edges(G,pos,alpha=0.1)
    # nx.draw_networkx_edges(G,pos,edgelist=edges,alpha=0.2)
    nx.draw_networkx_edges(G, pos, local, alpha=0.3, edge_color="green")
    nx.draw_networkx_edges(G, pos, bknet, alpha=0.8, edge_color="blue")
    nx.draw_networkx_edges(G, pos, tree, width=2, edge_color="blue")

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="green", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[median], node_size=150, node_color="black")
    nx.draw_networkx_nodes(G, pos, nodelist=backbone, node_size=50, node_color="red")

    # Draw node and edge labels
    if edisp:
        elabels = {e: ch[mesh.index(e)] for e in mesh}
        nx.draw_networkx_edge_labels(G, pos, elabels, font_size=10, font_color="grey")
    if labels:
        # Backbone nodes
        nLabel = {n: labels[n] for n in backbone}
        npos = {n: (pos[n][0], pos[n][1] + 0.03) for n in pos}
        nx.draw_networkx_labels(G, npos, nLabel, font_size=10, font_color="black")

        # All other nodes
        nLabel = {n: labels[n] for n in range(numNodes) if n not in backbone}
        npos = {n: (pos[n][0], pos[n][1] + 0.03) for n in pos}
        nx.draw_networkx_labels(G, npos, nLabel, font_size=10, font_color="gray")

    return plt
