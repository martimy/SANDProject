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


import math
import numpy as np
from .main import SANDAlgorithm

class MENTOR(SANDAlgorithm):
    def __init__(self):
        SANDAlgorithm.__init__(self)

    def run(
        self, cost, req, wparm=0, rparm=0.5, dparm=0.5, alpha=0.5, cap=1, slack=0.4
    ):
        self.nt = len(cost)  # Number of nodes
        self.cost = np.array(cost)  # Cost matrix (nt x nc)
        self.req = np.array(req)  # Traffic matrix (nt x nc)
        self.wparm = wparm  # fraction of max weight
        self.backbone = []
        self.maxWeight = 0
        self.assoc = []
        self.rParm = rparm  # fraction of max distance [0,1]
        self.dParm = dparm  # fraction for fig_of_merit [0,1]
        self.alpha = alpha  # PrimDijk parameter [0,1]
        self.cap = cap  # single-channel usable capacity
        self.slack = slack

        self.logger.debug("Starting MENTOR Algorithm")

        # PART 1 : find backbone nodes
        self.backbone, weight, Cassoc = self.__findBackbone()
        self.logger.debug(
            "Backbone nodes = {} : {}".format(
                len(self.backbone), ",".join(map(str, self.backbone))
            )
        )

        # PART 2 : Create topology
        median = self.__findBackboneMedian(self.backbone, weight)
        self.logger.debug("Backbone Median = {}".format(median))

        pred = self.__findPrimDijk(median, Cassoc)
        self.logger.debug(
            "Pred nodes = {} {}".format(len(pred), ",".join(map(str, pred)))
        )

        spPred, spDist = self.__setDist(median, pred)
        seqList, home = self.__setSequence(spPred)
        endList, multList = self.__compress(seqList, home)

        tree = [
            (i, pred[i])
            for i in range(len(pred))
            if i in self.backbone and i != pred[i]
        ]

        return {
            "backbone": self.backbone,
            "tree": tree,
            "mesh": endList,
            "channels": multList,
            "median": median,
        }

    # Set node weights
    def __findWeight(self):
        return np.sum(self.req + self.req.T, axis=1)

    # Find the Median node for all nodes
    def __findMedian(self, weight):
        return np.argmin(np.dot(self.cost, weight))

    # Find the Median node for backbone nodes
    def __findBackboneMedian(self, backbone, weight):
        return backbone[np.argmin(np.dot(self.cost[backbone], weight[backbone]))]

    # Select backbone nodes by comparing total traffic requirements
    # to a threshold
    def __findBackbone(self):
        weight = self.__findWeight()
        median = self.__findMedian(weight)
        self.maxWeight = np.max(weight)
        self.wparm *= self.maxWeight
        tbAssigned = np.arange(self.nt)
        backbone = tbAssigned[weight >= self.wparm]
        tbAssigned = tbAssigned[weight < self.wparm]

        radius = np.max(self.cost) * self.rParm
        Cassoc = np.arange(self.nt)

        while tbAssigned.size > 0:
            unassigned = np.array([])
            for c in tbAssigned:
                closest_backbone = np.argmin(self.cost[c, backbone])
                if self.cost[c, backbone[closest_backbone]] < radius:
                    Cassoc[c] = backbone[closest_backbone]
                else:
                    unassigned = np.append(unassigned, c)

            if unassigned.size == 0:
                break

            tbAssigned = unassigned
            merit = self.dParm * (self.cost[tbAssigned, median] / np.max(self.cost)) + (
                1 - self.dParm
            ) * (weight[tbAssigned] / self.maxWeight)
            n = tbAssigned[np.argmax(merit)]
            backbone = np.append(backbone, n)
            tbAssigned = np.setdiff1d(tbAssigned, n)

        return backbone, weight, Cassoc

    # Build initial tree topology
    def __findPrimDijk(self, root, Cassoc):
        assert root in self.backbone
        outTree = np.array(self.backbone)
        pred = np.array([root if Cassoc[x] == x else Cassoc[x] for x in range(self.nt)])
        inTree = []
        label = np.array(self.cost[root])
        while outTree.size > 0:
            n = root
            leastCost = np.inf
            for b in self.backbone:
                if label[b] < leastCost:
                    leastCost = label[b]
                    n = b
            inTree.append(n)
            outTree = outTree[outTree != n]
            label[n] = np.inf
            for o in outTree:
                x = self.alpha * leastCost + self.cost[o, n]
                label[o] = np.minimum(label[o], x)
                pred[o] = n
        return pred

    # Find the shortest path through the tree topology
    def __setDist(self, root, pred):
        preOrder = [root]
        n = 1
        while n < self.nt:
            for i in range(self.nt):
                if i not in preOrder and pred[i] in preOrder:
                    preOrder.append(i)
                    n += 1
        spDist = np.zeros((self.nt, self.nt))
        for i in range(self.nt):
            j = preOrder[i]
            p = pred[j]
            for k in range(i):
                l = preOrder[k]
                spDist[j, l] = spDist[l, j] = spDist[p, l] + self.cost[j, p]
        spPred = np.array([[pred[j] for j in range(self.nt)] for i in range(self.nt)])
        for i in range(self.nt):
            spPred[i, i] = i
        for i in range(self.nt):
            if i == root:
                continue
            p = pred[i]
            spPred[i, p] = i
            while p != root:
                pp = pred[p]
                spPred[i, pp] = p
                p = pp
        return spPred, spDist

    # Find the order in which to consider node pairs
    def __setSequence(self, spPred):
        home = np.full((self.nt, self.nt), None)
        pair = np.array([
            [i, j]
            for i in range(self.nt)
            for j in range(i + 1, self.nt)
        ])
        npairs = len(pair)
        nDep = np.zeros(npairs, dtype=int)
        dep1 = np.zeros(npairs, dtype=int)
        dep2 = np.zeros(npairs, dtype=int)
        for p in range(npairs):
            pr = pair[p]
            i, j = pr
            p1 = spPred[i, j]
            p2 = spPred[j, i]
            if p1 == i:
                h = None
            elif p1 == p2:
                h = p1
            else:
                if (self.cost[i, p1] + self.cost[p1, j]) <= (
                    self.cost[i, p2] + self.cost[p2, j]
                ):
                    h = p1
                else:
                    h = p2
            home[i, j] = h
            if h:
                pair_ih = i * self.nt + h
                dep1[p] = pair_ih
                nDep[pair_ih] += 1
                pair_jh = j * self.nt + h
                dep2[p] = pair_jh
                nDep[pair_jh] += 1
            else:
                dep1[p] = dep2[p] = None

        seqList = pair[nDep == 0]
        nseq = len(seqList)
        iseq = 0
        while iseq < nseq:
            p = seqList[iseq]
            iseq += 1
            d = dep1[p]
            if d:
                if nDep[d] == 1:
                    seqList = np.append(seqList, d)
                    nseq += 1
                else:
                    nDep[d] -= 1
            d = dep2[p]
            if d:
                if nDep[d] == 1:
                    seqList = np.append(seqList, d)
                    nseq += 1
                else:
                    nDep[d] -= 1
        return seqList, home

    # Select links and channels
    def __compress(self, seqList, home):
        reqList = np.array(self.req)
        npairs = (self.nt * (self.nt - 1)) // 2
        endList = []
        multList = []
        for p in range(npairs):
            x, y = seqList[p]
            h = home[x, y]
            mult = 0
            load = max(reqList[x, y], reqList[y, x])
            if load >= self.cap:
                mult = math.floor(load / self.cap)
                load -= mult * self.cap
            ovflow12 = ovflow21 = 0
            if (h is None and load > 0) or (load >= (1 - self.slack) * self.cap):
                mult += 1
            else:
                ovflow12 = max(0, reqList[x, y] - mult * self.cap)
                ovflow21 = max(0, reqList[y, x] - mult * self.cap)
            if mult > 0:
                endList.append((x, y))
                multList.append(mult)
            if ovflow12 > 0:
                reqList[x, h] += ovflow12
                reqList[h, y] += ovflow12
            if ovflow21 > 0:
                reqList[y, h] += ovflow21
                reqList[h, x] += ovflow21
        return endList, multList

    def __makePair(self, n, i, j):
        if i < j:
            return n * i + j
        else:
            return n * j + i

    def __splitPair(self, n, p):
        return p // n, p % n


