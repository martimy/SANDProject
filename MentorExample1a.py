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


# This is an example of using the MENTOR Algorithm in designing a backbone
# network based on an example from:
# Aaron Kershenbaum. 1993. Telecommunications Network Design Algorithms.
# McGraw-Hill, Inc., New York, NY, USA.


from sand.mentor_algo import MENTOR
from sand.mentor import printCost, plotNetwork
import random

# random.seed(5) #5

numNodes = 10
# Node labels:
labels = ["NYK", "LSA", "CHI", "HOU", "PHL", "DET", "SDG", "DAL", "SAN", "PHX"]

# Node positions:
pos = {
    0: (24, 12),
    1: (0, 4),
    2: (15, 11),
    3: (12, 2),
    4: (22, 11),
    5: (18, 13),
    6: (1, 2),
    7: (10, 4),
    8: (10, 0),
    9: (5, 2),
}

# Set cost matrix:
for p in pos:
    pos[p] = (pos[p][0] / 24, pos[p][1] / 24)
cost = [
    [1000, 8308, 3130, 5242, 1231, 2440, 8275, 5101, 5731, 7396],
    [8308, 1000, 6199, 5104, 8137, 6910, 1309, 4699, 4597, 2062],
    [3130, 6199, 1000, 3802, 2989, 1708, 6187, 3394, 4129, 5332],
    [5242, 5104, 3802, 1000, 5017, 4294, 4927, 1672, 1567, 4042],
    [1231, 8137, 2989, 5017, 1000, 2320, 8098, 4888, 5509, 7213],
    [2440, 6910, 1708, 4294, 2320, 1000, 6895, 3976, 4690, 6037],
    [8275, 1309, 6187, 4927, 8098, 6895, 1000, 4558, 4402, 1915],
    [5101, 4699, 3394, 1672, 4888, 3976, 4558, 1000, 1747, 3646],
    [5731, 4597, 4129, 1567, 5509, 4690, 4402, 1747, 1000, 3538],
    [7396, 2062, 5332, 4042, 7213, 6037, 1915, 3646, 3538, 1000],
]

# Set traffic requirements matrix:
req = [[random.randint(1, 10) for i in range(numNodes)] for j in range(numNodes)]
for i in range(len(req)):
    req[i][i] = 0

# Call MENTOR algorithm:
algo = MENTOR()
out = algo.run(cost, req, wparm=0.8, rparm=0.5, dparm=0.5, alpha=0.0, cap=32, slack=0.2)

# Print results:
printCost(out, cost, labels)
plotNetwork(out, pos, labels, title="MENTOR Algorthim - Example #1")
