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

cost = [
    [36, 0, 15, 78, 45, 65, 87, 80, 26, 19, 68],
    [46, 15, 0, 80, 55, 58, 82, 72, 31, 30, 72],
    [109, 78, 80, 0, 46, 45, 38, 48, 52, 62, 13],
    [67, 45, 55, 46, 0, 64, 74, 76, 26, 26, 33],
    [102, 65, 58, 45, 64, 0, 24, 14, 48, 61, 48],
    [122, 87, 82, 38, 74, 24, 0, 15, 65, 78, 47],
    [116, 80, 72, 48, 76, 14, 15, 0, 61, 75, 54],
    [58, 26, 31, 52, 26, 48, 65, 61, 0, 13, 41],
    [46, 19, 30, 62, 26, 61, 78, 75, 13, 0, 51],
    [97, 68, 72, 13, 33, 48, 47, 54, 41, 51, 0],
]

nt = len(cost)
nc = len(cost[0])

# Center node:
ncenter = 0

# Set cost to center:
Ccost = [50] * (nc + 1)
Ccost[ncenter] = 0

# Set weight requirements:
weight = [1] * nt

algo = ADD()
out = algo.run(cost, Ccost, weight)

# Print results
printCost(out, cost)
