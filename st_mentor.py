#!/usr/bin/python3

# Copyright (c) 2017-2024 Maen Artimy
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


# This file includes a Python implementation of the MENTOR Algorithm described
# in: Aaron Kershenbaum. 1993. Telecommunications Network Design Algorithms.
# McGraw-Hill, Inc., New York, NY, USA.


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sand.mentor import MENTOR, printCost

WPARM = "Help"
RPARM = "Help"
DPARM = "Help"
ALPHA = "Help"
CAP = "Help"
SLACK = "Help"


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


def plotNetwork(out, numNodes, labels=[], edisp=True, title="MENTOR Algorithm"):
    mesh = out["mesh"]
    ch = out["channels"]
    backbone = out["backbone"]
    median = out["median"]
    tree = out["tree"]

    # Separate the mesh
    bknet = [p for p in mesh if p[0] in backbone and p[1] in backbone]
    local = [p for p in mesh if p not in bknet]

    plt.figure(figsize=(6, 6), facecolor="white")
    G = nx.path_graph(numNodes)
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
        nLabel = {n: labels[n] for n in backbone}
        npos = {n: (pos[n][0], pos[n][1] + 0.03) for n in pos}
        nx.draw_networkx_labels(G, npos, nLabel, font_size=10, font_color="black")

    return plt


def main():
    st.title("MENTOR Algorithm Example")

    st.sidebar.header("Upload Cost Matrix")
    uploaded_cost_file = st.sidebar.file_uploader("Upload Cost CSV", type=["csv"])

    st.sidebar.header("Upload Requirements Matrix")
    uploaded_req_file = st.sidebar.file_uploader(
        "Upload Requirements CSV", type=["csv"]
    )

    st.sidebar.header("Input Paramters")
    wparm = st.sidebar.slider(
        "W Parameter", min_value=0.1, max_value=2.0, step=0.1, value=1.0, help=WPARM
    )
    rparm = st.sidebar.slider(
        "R Parameter", min_value=0.1, max_value=1.0, step=0.1, value=0.5, help=RPARM
    )
    dparm = st.sidebar.slider(
        "D Parameter", min_value=0.1, max_value=1.0, step=0.1, value=0.5, help=DPARM
    )
    alpha = st.sidebar.slider(
        "Alpha", min_value=0.0, max_value=1.0, step=0.1, value=0.0, help=ALPHA
    )
    # cap = st.sidebar.slider("Capacity", min_value=16, max_value=64, step=4, value=32, help=CAP)
    slack = st.sidebar.slider(
        "Slack", min_value=0.0, max_value=1.0, step=0.1, value=0.2, help=SLACK
    )
    cap = st.sidebar.number_input("Capacity", min_value=1, help="CAP")

    if uploaded_cost_file and uploaded_req_file:
        cost_df = pd.read_csv(uploaded_cost_file)
        req_df = pd.read_csv(uploaded_req_file)

        # Check if the CSV file has the same number of columns and rows
        if cost_df.shape[0] != cost_df.shape[1]:
            st.error(
                "Error: The number of rows and columns in the CSV file must be equal."
            )
            return

        # Check if the requirements CSV file has the same number of columns and rows as the cost matrix
        if req_df.shape != cost_df.shape:
            st.error(
                "Error: The requirements matrix CSV file must have the same dimensions as the cost matrix."
            )
            return

        cost = cost_df.values.tolist()

        st.subheader("Design Input")
        expand_cost = st.expander("Cost Matrix")
        expand_cost.table(cost_df)

        req = req_df.values.tolist()

        expand_req = st.expander("Requirements Matrix:")
        expand_req.table(req_df)

        # Call MENTOR algorithm:
        algo = MENTOR()
        out = algo.run(
            cost,
            req,
            wparm=wparm,
            rparm=rparm,
            dparm=dparm,
            alpha=alpha,
            cap=cap,
            slack=slack,
        )

        # Print results:
        st.subheader("Design Output")
        labels = list(cost_df.columns)

        df, total_cost, num_backbone, num_links = cost_to_dataframe(out, cost, labels)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cost", total_cost)
        col2.metric("Number of backbone nodes", num_backbone)
        col3.metric("Number of backbone links", num_links)

        st.subheader("Network Plot:")
        st.dataframe(df)

        # Plot Network:
        st.subheader("Network Plot:")
        plt = plotNetwork(out, len(cost), labels, title="MENTOR Algorithm - Example")
        plt.axis("off")
        st.pyplot(plt)

    else:
        st.warning("Upload input files")


if __name__ == "__main__":
    main()
