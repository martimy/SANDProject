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

import logging
from sand.mentor_algo import MENTOR
from sand.mentor import printCost, plotNetwork
import random

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

random.seed(5)  # 5

numNodes = 5
# Node labels:
labels = ["A", "B", "C", "D", "E"]

# Node positions:
pos = {
    0: (0, 1),
    1: (1, 1),
    2: (0.5, 0.5),
    3: (0, 0),
    4: (1, 0.5),
}

# Set cost matrix:
cost = [
[1,8,4,8,8],
[8,1,4,8,2],
[4,4,1,4,4],
[8,8,4,1,8],
[8,2,4,8,1]]

# Set traffic requirements matrix:
# req = [[random.randint(1, 10) for i in range(numNodes)] for j in range(numNodes)]
#req = [[8 for i in range(numNodes)] for j in range(numNodes)]
req = [
[0,8,18,8,8],
[8,0,18,8,8],
[18,18,0,18,8],
[8,8,18,0,8],
[8,8,8,8,0]]
for i in range(len(req)):
    req[i][i] = 0

# Call MENTOR algorithm:
algo = MENTOR()
algo.log("mentor.log")

out = algo.run(
    cost, req, wparm=0.5, rparm=0.5, dparm=0.5, alpha=0.0, cap=30, slack=0.2
)
# out = algo.run(cost, req, wparm=0, rparm=0.5, dparm=0.5, alpha=0.0, cap=32, slack=0.2)

# Print results:
printCost(out, cost, labels)
plotNetwork(out, pos, labels)
