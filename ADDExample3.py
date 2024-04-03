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

from sand.add import *
import math, random

numNodes = 250
# Node labels:
labels = [str(i) for i in range(numNodes)]

# Node positions:
pos = {i: (random.random(), random.random()) for i in range(numNodes)}

# Set cost matrix:
cost = [[2**32 - 1 for i in range(numNodes)] for i in range(numNodes)]
for i in range(numNodes):
    for j in range(i, numNodes):
        r1, c1 = pos[i]
        r2, c2 = pos[j]
        d = round(math.sqrt(abs(r1 - r2) + abs(c1 - c2)) * 1000)
        cost[i][j] = cost[j][i] = d

# Locate the centre node near the centre of the graph:
dmin = 2**32 - 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# Set cost to centre:
conc_cost = 100
Ccost = [cost[i][ncenter] + conc_cost for i in range(numNodes)]
Ccost[ncenter] = 0

# Set weight requirements:
weight = [1] * numNodes

# Call ADD algorithm:
algo = ADD()
out = algo.run(cost, Ccost, weight, Wlimit=5, center=ncenter)

# Print results
printCost(out, cost)
plotNetwork(out, pos, labels, title="ADD Algorithm - Example 3")
