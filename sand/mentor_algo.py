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
    INF = 2**16 - 1

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
     
        self.logger.debug(f"Backbone nodes = {self.backbone}")
        self.logger.debug(f"Associations   = {Cassoc}")
        self.logger.debug(f"Weight = {weight}")
        
        # PART 2 : Create topology
        median = self.__findBackboneMedian(self.backbone, weight)
        self.logger.debug(f"Backbone Median = {median}")

        pred = self.__findPrimDijk(median, Cassoc)
        self.logger.debug(f"Pred nodes     = {','.join(map(str, pred))}")

        spPred, spDist = self.__setDist(median, pred)
        self.logger.debug(f"spPred nodes = \n{spPred}")
        self.logger.debug(f"spDist = \n{spDist}")

        seqList, home = self.__setSequence(spPred)
        self.logger.debug(f"seqList = \n{seqList}")
        self.logger.debug(f"Home = \n{np.array(home)}")
        
        endList, multList = self.__compress(seqList, home)
        self.logger.debug(f"endList = {endList}")
        self.logger.debug(f"multList = {multList}")
        
        tree = [
            (i, pred[i])
            for i in range(len(pred))
            if i in self.backbone and i != pred[i]
        ]
        self.logger.debug(f"tree = {tree}")
        
        return {
            "backbone": self.backbone,
            "tree": tree,
            "mesh": endList,
            "channels": multList,
            "median": median,
        }

    # Set node weights
    def __findWeights(self):
        weights = np.zeros(self.nt)  # Initialize weights array with zeros
        for n in range(self.nt):
            weights[n] = np.sum(self.req[n]) + np.sum(self.req[:, n])
        return weights

    # Find the Median node for all nodes
    def __findMedian(self, weights):
        """
        Return the index of the node with the minimum moment

        """
        moments = np.sum(self.cost * weights, axis=1)
        return np.argmin(moments)

    # Find the Median node for backbone nodes
    def x__findBackboneMedian(self, backbone, weight):
        moment = []
        for i in range(len(backbone)):
            cw = [self.cost[backbone[i]][j] * weight[j] for j in backbone]
            moment.append(sum(cw))
        return backbone[moment.index(min(moment))]

    def __findBackboneMedian(self, backbone, weight):
        # Initialize moments array with zeros
        moments = np.zeros(len(backbone))
        for i, node in enumerate(backbone):
            # Vectorized computation of cost * weight
            cw = self.cost[node][backbone] * weight[backbone]
            moments[i] = np.sum(cw)
        # Return the node with the minimum moment
        return backbone[np.argmin(moments)]

    # Select backbone nodes by comparing total traffic requirements
    # to a threshold
    def __findBackbone(self):
        weights = self.__findWeights()
        median = self.__findMedian(weights)

        self.maxWeight = np.max(weights)
        weight_threshold = self.wparm * self.maxWeight

        # Find indices where weights meet threshold. These are backbone nodes
        backbone = np.where(weights >= weight_threshold)[0]

        # Remaining nodes are to be decided later
        tbAssigned = np.where(weights < weight_threshold)[0]

        # Calculate the distance matrix between tbAssigned and backbone nodes
        distances = self.cost[tbAssigned][:, backbone]

        # Initialize Cassoc with indices of backbone nodes closest to each
        # tbAssigned node
        closest_backbone = np.argmin(distances, axis=1)

        # These are local nodes
        Cassoc = np.arange(self.nt)
        Cassoc[tbAssigned] = np.array(backbone)[closest_backbone]

        # find the maximum distance (radius) between any two nodes
        self.maxDist = np.max(self.cost)

        # Determine which nodes need further evaluation
        radius = self.maxDist * self.rParm
        need_evaluation = np.any(distances >= radius, axis=1)
        unassigned = tbAssigned[need_evaluation]

        # for the remaining nodes:
        # calculate the distance between each unassigned node and
        # all backbone nodes to determine if it needs to be assigned

        # The Figure of Merit function
        def figMerit(u):
            return self.dParm * (self.cost[u][median] / self.maxDist) + (
                1 - self.dParm
            ) * (weights[u] / self.maxWeight)

        while unassigned.size > 0:
            # The node with the maximum Figure of Merit is added to backbone
            merit = np.array([figMerit(u) for u in unassigned])
            n = unassigned[np.argmax(merit)]

            # Add the selected node to the backbone
            backbone = np.append(backbone, n)
            unassigned = np.setdiff1d(unassigned, [n])

            for m in unassigned:
                if self.cost[m, n] < radius:
                    Cassoc[m] = n
                    unassigned = np.setdiff1d(unassigned, [m])

            # Find indices of unassigned nodes within radius
            within_radius = np.where(self.cost[unassigned, n] < radius)[0]

            # Update Cassoc for nodes within radius
            Cassoc[unassigned[within_radius]] = n
            # Remove assigned nodes from unassigned
            unassigned = unassigned[~within_radius]

        backbone.sort()
        return backbone, weights, Cassoc

    # Build initial tree topology
    def __findPrimDijk(self, root, Cassoc):
        assert root in self.backbone

        outTree = self.backbone
        pred = np.array([root if Cassoc[i] == i else Cassoc[i] for i in range(self.nt)])
        inTree = []
        label = self.cost[root]
        while outTree.size > 0:
            n = root
            leastCost = MENTOR.INF
            for b in self.backbone:
                if label[b] < leastCost:
                    leastCost = label[b]
                    n = b

            inTree.append(n)
            outTree = outTree[outTree != n]  # Remove n from outTree
            label[n] = MENTOR.INF
            for o in outTree:
                x = self.alpha * leastCost + self.cost[o][n]
                if label[o] > x:
                    label[o] = x
                    pred[o] = n

        return pred

    # Find the shortest path through the tree topology
    def __setDist(self, root, pred):
        # Initialize preOrder array with root
        preOrder = [root]

        # Traverse the tree and populate preOrder array
        n = 1
        while n < self.nt:
            for i in range(self.nt):
                if (i not in preOrder) and (pred[i] in preOrder):
                    preOrder.append(i)
                    n += 1

        # Initialize spDist and spPred arrays
        spDist = np.zeros((self.nt, self.nt))

        # Find the distance (cost) of the shortest path between any two nodes
        # along the backbone tree
        for i in range(self.nt):
            j = preOrder[i]
            p = pred[j]
            # spDist[j][j] = 0
            for k in range(i):
                l = preOrder[k]
                spDist[j][l] = spDist[l][j] = spDist[p][l] + self.cost[j][p]

        # Set the predecessors
        spPred = np.tile(pred, (self.nt, 1))
        for i in range(self.nt):
            spPred[i][i] = i

        for i in range(self.nt):
            if i == root:
                continue
            p = pred[i]
            spPred[i][p] = i
            while p != root:
                pp = pred[p]
                spPred[i][pp] = p
                p = pp

        return spPred, spDist

    # Find the order in which to consider node pairs
    def __setSequence(self, spPred):
        home = [[None for i in range(self.nt)] for j in range(self.nt)]

        # make pairs
        pair = [
            self.__makePair(self.nt, i, j)
            for i in range(self.nt)
            for j in range(i + 1, self.nt)
        ]

        nn = self.nt * self.nt
        nDep = [0] * nn
        dep1 = [0] * nn
        dep2 = [0] * nn
        for p in range(len(pair)):
            pr = pair[p]
            i, j = self.__splitPair(self.nt, pr)
            p1 = spPred[i][j]
            p2 = spPred[j][i]
            if p1 == i:  # this is a tree link
                h = None
            elif p1 == p2:  # 2-hop path, only one possible home
                h = p1
            else:
                if (self.cost[i][p1] + self.cost[p1][j]) <= (
                    self.cost[i][p2] + self.cost[p2][j]
                ):
                    h = p1
                else:
                    h = p2
            home[i][j] = h
            if h:
                # increment the number of pairs that depend on (i, h)
                pair_ih = self.__makePair(self.nt, i, h)
                dep1[pr] = pair_ih
                nDep[pair_ih] += 1
                pair_jh = self.__makePair(self.nt, j, h)
                dep2[pr] = pair_jh
                nDep[pair_jh] += 1
            else:
                dep1[pr] = dep2[pr] = None

        seqList = [p for p in pair if nDep[p] == 0]

        nseq = len(seqList)
        iseq = 0
        while iseq < nseq:
            p = seqList[iseq]
            iseq += 1
            d = dep1[p]
            if d:
                if nDep[d] == 1:
                    seqList.append(d)
                    nseq += 1
                else:
                    nDep[d] -= 1

            d = dep2[p]
            if d:
                if nDep[d] == 1:
                    seqList.append(d)
                    nseq += 1
                else:
                    nDep[d] -= 1

        return seqList, home

    # Select links and channels
    def __compress(self, seqList, home):
        # copy req to reqList
        reqList = list(self.req)
        for row in range(len(self.req)):
            reqList[row] = list(self.req[row])

        npairs = (self.nt * (self.nt - 1)) // 2
        endList = []
        multList = []

        for p in range(npairs):
            x, y = self.__splitPair(self.nt, seqList[p])
            h = home[x][y]

            # assume full duplex always
            mult = 0
            load = max([reqList[x][y], reqList[y][x]])
            if load >= self.cap:
                mult = math.floor(load / self.cap)
                load -= mult * self.cap

            ovflow12 = ovflow21 = 0
            if (h is None and load > 0) or (load >= (1 - self.slack) * self.cap):
                mult += 1
            else:
                ovflow12 = max([0, reqList[x][y] - mult * self.cap])
                ovflow21 = max([0, reqList[y][x] - mult * self.cap])

            if mult > 0:
                endList.append((x, y))
                multList.append(mult)

            if ovflow12 > 0:
                reqList[x][h] += ovflow12
                reqList[h][y] += ovflow12
            if ovflow21 > 0:
                reqList[y][h] += ovflow21
                reqList[h][x] += ovflow21

        return endList, multList

    def __makePair(self, n, i, j):
        return np.where(i < j, n * i + j, n * j + i)

    def __splitPair(self, n, p):
        return p // n, p % n
