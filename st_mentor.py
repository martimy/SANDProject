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


import streamlit as st
import pandas as pd
from sand.mentor_algo import MENTOR
from sand.mresults import cost_to_dataframe, plot_network

WPARM = "The weight threshold for selecting the backbone nodes"
RPARM = "The cost threshold for selecting the local nodes"
DPARM = "The paramter controlling the relative importance of weight vs cost \
         in selecting backbone nodes"
ALPHA = "The paramter controls the backbone shape. Value of 0 yields a \
         minimum spanning tree (MST) and the value of 1 yields a star."
CAP = "The usable capacity of a channel"
SLACK = "The paramter controls the threshold of selecting direct links"

ABOUT = r"""
This application is an implementation of the MENTOR algorithm. 
The MENTOR (MEsh Network Topology Optimization and Routing) algorithm was 
developed by Aaron Kershenbaum, Parviz Kermani, and George A. Grove in 1991 
to design mesh networks, focusing on their initial topology.

The algorithm assumes three conditions for achieving a low-cost topology:

   - Traffic is routed on direct paths.
   - Links are sufficiently utilized without being highly utilized.
   - High capacity links are used.

Routing Strategy:

   - For large traffic requirements, the algorithm sends traffic over a direct 
   route between the source and destination. This satisfies all three 
   conditions.
   - In other cases, traffic is sent via a path within a tree. 
   The algorithm aggregates traffic as much as possible, ensuring at least the 
   last two conditions are met.
   - The topology over which traffic flows is defined using Dijkstra's and 
   Prim's algorithms.
"""


def main():
    st.title("Backbone Design using MENTOR")
    expand_about = st.expander("About", expanded=False)
    expand_about.markdown(ABOUT)

    st.sidebar.header("Upload Requirements Matrix")
    uploaded_req_file = st.sidebar.file_uploader(
        "Upload Requirements CSV", type=["csv"]
    )

    st.sidebar.header("Upload Cost Matrix")
    uploaded_cost_file = st.sidebar.file_uploader("Upload Cost CSV", type=["csv"])

    st.sidebar.header("Upload Image")
    uploaded_img_file = st.sidebar.file_uploader("Choose a file")

    st.sidebar.header("Input Paramters")
    wparm = st.sidebar.slider(
        "W Parameter", min_value=0.0, max_value=1.0, step=0.1, value=1.0, help=WPARM
    )
    rparm = st.sidebar.slider(
        "R Parameter", min_value=0.0, max_value=1.0, step=0.1, value=0.5, help=RPARM
    )
    dparm = st.sidebar.slider(
        "D Parameter", min_value=0.0, max_value=1.0, step=0.1, value=0.5, help=DPARM
    )
    alpha = st.sidebar.slider(
        "Alpha", min_value=0.0, max_value=1.0, step=0.1, value=0.0, help=ALPHA
    )
    slack = st.sidebar.slider(
        "Slack", min_value=0.0, max_value=1.0, step=0.1, value=0.2, help=SLACK
    )
    cap = st.sidebar.number_input(
        "Capacity", min_value=1, step=1000, value=1000000, help=CAP
    )

    if uploaded_cost_file and uploaded_req_file:
        cost_df = pd.read_csv(uploaded_cost_file)
        req_df = pd.read_csv(uploaded_req_file)

        # Check if the CSV file has the same number of columns and rows
        if cost_df.shape[0] != cost_df.shape[1]:
            st.error(
                "Error: The number of rows and columns in the CSV file must be equal."
            )
            return

        # Check if the requirements CSV file has the same number of columns
        # and rows as the cost matrix
        if req_df.shape != cost_df.shape:
            st.error(
                "Error: The requirements matrix CSV file must have the same \
                    dimensions as the cost matrix."
            )
            return

        st.subheader("Design Input")
        expand_cost = st.expander("Cost Matrix")
        cost_df = expand_cost.data_editor(
            cost_df, use_container_width=True, hide_index=True
        )

        expand_req = st.expander("Requirements Matrix:")
        req_df = expand_req.data_editor(
            req_df, use_container_width=True, hide_index=True
        )

        cost = cost_df.values.tolist()
        req = req_df.values.tolist()

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

        st.subheader("Network Data:")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Plot Network:
        st.subheader("Network Plot:")
        if uploaded_img_file is not None:
            bytes_data = uploaded_img_file.getvalue()
            plt = plot_network(out, len(cost), labels, image=bytes_data)
        else:
            plt = plot_network(out, len(cost), labels)

        plt.axis("off")
        st.pyplot(plt)

    else:
        st.warning("Upload input files")


if __name__ == "__main__":
    main()
