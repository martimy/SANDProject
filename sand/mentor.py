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


# This file includes a Python implementation of the MENTOR Algorithm described
# in: Aaron Kershenbaum. 1993. Telecommunications Network Design Algorithms.
# McGraw-Hill, Inc., New York, NY, USA.


from .main import SANDAlgorithm
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class MENTOR(SANDAlgorithm):
    def __init__(self):
        SANDAlgorithm.__init__(self)

    def run(self, cost, req, wparm=0, rparm=0.5, dparm=0.5, alpha=0.5, cap=1,
            slack=0.4):
        self.nt = len(cost)                 # Number of nodes
        self.cost = cost                    # Cost matrix (nt x nc)
        self.req = req                      # Traffic matrix (nt x nc)
        self.wparm = wparm                  # fraction of max weight
        self.backbone = []
        self.maxWeight = 0
        self.assoc = []
        self.rParm = rparm                  # fraction of max distance [0,1]
        self.dParm = dparm                  # fraction for fig_of_merit [0,1]
        self.alpha = alpha                  # PrimDijk parameter [0,1]
        self.cap = cap                      # single-channel usable capacity
        self.slack = slack

        self.logger.debug('Starting MENTOR Algorithm')

        # PART 1 : find backbone nodes
        self.backbone, weight, Cassoc = self.__findBackbone()
        self.logger.debug('Backbone nodes = {} : {}'.format(len(self.backbone),
                                                            ','.join(map(str, self.backbone))))

        # PART 2 : Create topology
        median = self.__findBackboneMedian(self.backbone, weight)
        self.logger.debug('Backbone Median = {}'.format(median))

        pred = self.__findPrimDijk(median, Cassoc)
        self.logger.debug('Pred nodes = {} {}'.format(
            len(pred), ','.join(map(str, pred))))

        spPred, spDist = self.__setDist(median, pred)
        seqList, home = self.__setSequence(spPred)
        endList, multList = self.__compress(seqList, home)

        tree = [(i, pred[i]) for i in range(len(pred))
                if i in self.backbone and i != pred[i]]
        return {"backbone": self.backbone, "tree": tree, "mesh": endList,
                "channels": multList, "median": median}

    # Set node weights
    def __findWeight(self):
        weight = []                     # This is the weight
        for n in range(self.nt):
            sum = 0
            for i in range(self.nt):
                sum += self.req[n][i]
                sum += self.req[i][n]
            weight.append(sum)
        return weight

    # Find the Median node for all nodes
    def __findMedian(self, weight):
        moment = []
        for i in range(self.nt):
            cw = [self.cost[i][j] * weight[j] for j in range(self.nt)]
            moment.append(sum(cw))
        return moment.index(min(moment))

    # Find the Median node for backbone nodes
    def __findBackboneMedian(self, backbone, weight):
        moment = []
        for i in range(len(backbone)):
            cw = [self.cost[backbone[i]][j] * weight[j] for j in backbone]
            moment.append(sum(cw))
        return backbone[moment.index(min(moment))]

    # Select backbone nodes by comparing total traffic requirements
    # to a threshold
    def __findBackbone(self):
        backbone = []
        weight = self.__findWeight()
        median = self.__findMedian(weight)

        self.maxWeight = max(weight)
        self.wparm *= self.maxWeight
        tbAssigned = []                 # to be assigned nodes
        for n in range(self.nt):
            if(weight[n] >= self.wparm):
                backbone.append(n)
            else:
                tbAssigned.append(n)

        # find the maximum distance (radius) between any two nodes
        self.maxDist = max([max(r) for r in self.cost])

        # for the remaining nodes:
        # calculate the distance between each unassigned node and
        # all backbone nodes to determine if it needs to be assigned

        # The Figure of Merit function
        def figMerit(u):
            return self.dParm * (self.cost[u][median] / self.maxDist) + \
                (1-self.dParm) * (weight[u] / self.maxWeight)

        radius = self.maxDist * self.rParm
        Cassoc = [i for i in range(self.nt)]
        while tbAssigned:
            # while there are nodes to be assigned, associate nodes with
            # the closest backbone node if there are within a given radius
            unassigned = []
            for c in tbAssigned:
                lowestR = self.INF                # any big +ve number
                assgnd = False
                for b in backbone:
                    if(self.cost[c][b] < radius):
                        # if the distance is lower than the radius,
                        # the node becomes a terminal node associated
                        # with the closest backbone (or cheapest to connect)
                        if(self.cost[c][b] < lowestR):
                            lowestR = self.cost[c][b]
                            Cassoc[c] = b
                            assgnd = True
                if not assgnd:
                    # This node needs further evaluation
                    unassigned.append(c)

            # Terminate the loop if there are no more unassigned nodes
            if not unassigned:
                break

            # Determine if a node can be a backbone node based on
            # Figure of Merit
            tbAssigned = unassigned
            merit = [figMerit(u) for u in tbAssigned]
            n = tbAssigned[merit.index(max(merit))]
            backbone.append(n)
            tbAssigned.remove(n)

        return backbone, weight, Cassoc

    # Build initial tree topology
    def __findPrimDijk(self, root, Cassoc):
        assert root in self.backbone
        #outTree = list(range(self.nt))
        outTree = list(self.backbone)
        # strating with the terminal association list, assigning all backbone nodes
        # to root as predecessor
        pred = [(lambda x: root if Cassoc[x] == x else Cassoc[x])(i)
                for i in range(self.nt)]
        inTree = []
        label = list(self.cost[root])  # copy the cost of every node to root
        while outTree:
            # select a node that is in the backbone, not already inTree,
            # and has the least cost
            # the first item selected will be the root
            n = root
            leastCost = self.INF
            for b in self.backbone:
                if label[b] < leastCost:
                    leastCost = label[b]
                    n = b

            #leastCost = min(label)
            #n = label.index(leastCost)
            inTree.append(n)
            outTree.remove(n)
            label[n] = self.INF  # prevent the node from being considered again
            for o in outTree:
                x = self.alpha * leastCost + self.cost[o][n]
                if(label[o] > x):
                    label[o] = x
                    pred[o] = n
        return pred

    # Find the shortest path through the tree topology
    def __setDist(self, root, pred):
        preOrder = [root]
        n = 1
        while n < self.nt:
            for i in range(self.nt):
                if((i not in preOrder) and (pred[i] in preOrder)):
                    preOrder.append(i)
                    n += 1

        # Find the distance (cost) of the shortest path between any two nodes
        # along the backbone tree
        spDist = [[0 for j in range(self.nt)] for i in range(self.nt)]
        for i in range(self.nt):
            j = preOrder[i]
            p = pred[j]
            #spDist[j][j] = 0
            for k in range(i):
                l = preOrder[k]
                spDist[j][l] = spDist[l][j] = spDist[p][l] + self.cost[j][p]

        # Set the predecessors
        spPred = [[pred[j] for j in range(self.nt)] for i in range(self.nt)]
        for i in range(self.nt):
            spPred[i][i] = i
        for i in range(self.nt):
            if(i == root):
                continue
            p = pred[i]
            spPred[i][p] = i
            while(p != root):
                pp = pred[p]
                spPred[i][pp] = p
                p = pp

        return spPred, spDist

    # Find the order in which to consider node pairs
    def __setSequence(self, spPred):
        home = [[None for i in range(self.nt)] for j in range(self.nt)]
        pair = [self.__makePair(self.nt, i, j) for i in range(self.nt)
                for j in range(i+1, self.nt)]

        np = self.nt * self.nt
        nDep = [0] * np
        dep1 = [0] * np
        dep2 = [0] * np
        for p in range(len(pair)):
            pr = pair[p]
            i, j = self.__splitPair(self.nt, pr)
            p1 = spPred[i][j]
            p2 = spPred[j][i]
            if(p1 == i):  # this is a tree link
                h = None
            elif (p1 == p2):  # 2-hop path, only one possible home
                h = p1
            else:
                if((self.cost[i][p1] + self.cost[p1][j])
                        <= (self.cost[i][p2] + self.cost[p2][j])):
                    h = p1
                else:
                    h = p2
            home[i][j] = h
            if(h):
                # increment the number of pairs that depend on (i, h)
                pair_ih = self.__makePair(self.nt, i, h)
                dep1[pr] = pair_ih
                nDep[pair_ih] += 1
                pair_jh = self.__makePair(self.nt, j, h)
                dep2[pr] = pair_jh
                nDep[pair_jh] += 1
            else:
                dep1[pr] = dep2[pr] = None

        #print("nDep :\n",nDep)
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

        #print("seqList :\n", seqList)

        return seqList, home

    # Select links and channels
    def __compress(self, seqList, home):
        # copy req to reqList
        reqList = list(self.req)
        for row in range(len(self.req)):
            reqList[row] = list(self.req[row])

        npairs = (self.nt * (self.nt - 1))//2
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
            if (h is None and load > 0) or (load >= (1-self.slack) * self.cap):
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
        if i < j:
            return n * i + j
        else:
            return n * j + i

    def __splitPair(self, n, p):
        return p//n, p % n

# print cost list of network produced by MENTOR algorithm


def printCost(out, cost, labels):
    backbone = out["backbone"]
    mesh = out["mesh"]
    chlist = out["channels"]

    total = 0
    print('%10s%10s%4s%8s' % ('From', 'To', 'Ch', 'Cost($)'))
    print(('=' * 34))
    for i in range(len(mesh)):
        x, y = mesh[i]
        total += cost[x][y]*chlist[i]
        print('%10s%10s%4d%8d' % (labels[x], labels[y], chlist[i],
                                  cost[x][y]*chlist[i]))
    print(('=' * 34))
    print('%12s%8d' % ('Total cost', total))
    print('Number of backbone nodes =', len(backbone))
    bknet = [p for p in mesh if p[0] in backbone and p[1] in backbone]
    print('Number of links in the backbone =', len(bknet))


# Plot topology produced by MENTOR algorithm
def plotNetwork(out, pos, labels=[], edisp=True, filename="figure_mentor.png",
                title='MENTOR Algorithm'):
    numNodes = len(pos)
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

    # nx.draw_networkx_edges(G,pos,alpha=0.1)
    # nx.draw_networkx_edges(G,pos,edgelist=edges,alpha=0.2)
    nx.draw_networkx_edges(G, pos, local, alpha=0.3, edge_color="green")
    nx.draw_networkx_edges(G, pos, bknet, alpha=0.8, edge_color="blue")
    nx.draw_networkx_edges(G, pos, tree, width=2, edge_color="blue")

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="green", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[median], node_size=150,
                           node_color="black")
    nx.draw_networkx_nodes(G, pos, nodelist=backbone, node_size=50,
                           node_color="red")

    # Draw node and edge labels
    if edisp:
        elabels = {e: ch[mesh.index(e)] for e in mesh}
        nx.draw_networkx_edge_labels(G, pos, elabels, edgelist=mesh,
                                     font_size=10, font_color="grey")
    if labels:
        nLabel = {n: labels[n] for n in backbone}
        npos = {n: (pos[n][0], pos[n][1]+0.03) for n in pos}
        nx.draw_networkx_labels(G, npos, nLabel, nodelist=backbone,
                                font_size=10, font_color="black")

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
