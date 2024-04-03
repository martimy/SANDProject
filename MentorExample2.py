#!/usr/bin/python3

# Copyright (c) 2017 Maen Artimy
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


# This file is part of the SAND library

from sand.mentor import *
import math, random

numNodes = 250
# Node labels:
labels = [str(i) for i in range(numNodes)]

# Node positions:
pos = {i: (random.random(), random.random()) for i in range(numNodes)}

# Set cost matrix:
base_cost = 1000
cost = [[2**32 - 1 for j in range(numNodes)] for i in range(numNodes)]
for i in range(numNodes):
    for j in range(i, numNodes):
        r1, c1 = pos[i]
        r2, c2 = pos[j]
        d = round(math.sqrt(abs(r1 - r2) + abs(c1 - c2)) * base_cost)
        cost[i][j] = cost[j][i] = d


# Set traffic requirements matrix:
base_cap = 1  # Mbps
req = [
    [random.random() * 10 * base_cap for j in range(numNodes)] for i in range(numNodes)
]

# Call MENTOR algorithm:
algo = MENTOR()
out = algo.run(
    cost,
    req,
    wparm=0.95,
    rparm=0.5,
    dparm=0.5,
    alpha=0.5,
    cap=1000 * base_cap,
    slack=0.2,
)

# Print results:
printCost(out, cost, labels)
plotNetwork(out, pos, labels, edisp=False, title="MENTOR Algorthim - Example #2")
