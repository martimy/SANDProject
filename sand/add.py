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


# This file includes a Python implementation of the ADD Algorithm described in: 
# Aaron Kershenbaum. 1993. Telecommunications Network Design Algorithms. McGraw-
# Hill, Inc., New York, NY, USA.


from .main import SANDAlgorithm
import networkx as nx
import matplotlib.pyplot as plt

class ADD(SANDAlgorithm):  
    def __init__(self):
        SANDAlgorithm.__init__(self)
        
    def run(self, cost, Ccost, weight, center=0, Wlimit=10, th_move=0):        
        self.nt = len(cost)                 # Number of terminals
        self.nc = len(cost[0])              # Number of concentrators
        self.Wlimit = Wlimit                # Maximum number of terminals
        self.th_move = th_move              # Cost to move terminal  
        self.cost = cost                    # Cost matrix (nt x nc)
        self.weight = weight                # Load vector (nt)
        self.Ccost = Ccost                  # Cost to build a concentrator (nc)
        self.center = center                # Index of the central location
 

        self.logger.debug('Starting ADD Algorithm')
        # Associate all nodes with the central location
        # and calculate the initial cost
        self.Cassoc = [self.center] * self.nt         # Association with a conc
        self.cTotal = sum([self.cost[t][self.Cassoc[t]] 
                            for t in range(self.nt)]) + self.Ccost[self.center]
        self.logger.debug("Initial cost = %d" % self.cTotal)

        # Calculate the cost savings for the remaining concentrators
        remConc = list(range(self.nc))
        remConc.remove(self.center)
        self.logger.debug("Concentrators to be evaluated = %s" % remConc)
        
        while len(remConc) > 0:
            savings = 0
            conc = 0
            for t in remConc:
                expense = self.__evalConc(t)
                if(expense < savings):
                    savings = expense
                    conc = t

            if(savings < 0):
                self.__addConc(conc)
                self.cTotal += savings
                self.logger.debug("Concentrator %d \
                         is added for total cost of %d" % (conc, self.cTotal))
                remConc.remove(conc)
                self.logger.debug("Current association = %s" % self.Cassoc)
            else:
                self.logger.debug("No more savings!")
                break
        
        # Sanity check
        tCost = sum([self.cost[t][self.Cassoc[t]] for t in range(self.nt)])
        cCost = sum([self.Ccost[c] for c in set(self.Cassoc)])
        if((tCost + cCost) != self.cTotal):
            self.logger.error("Something is wrong, \
            detected cost discrepancy! %d %d %d" % (tCost, cCost, self.cTotal))
            
        return({"cost": self.cTotal, "center": self.center, "num": self.nt,
                "assoc": self.Cassoc, "conc":set(self.Cassoc)})

    def __evalConc(self, c):
        delta = [0] * self.nt
        ter = [0] * self.nt
        
        expense = self.Ccost[c]
        slack = self.Wlimit
        n = 0
        # Calculates the saving if any that results from 
        # connecting terminal t to concentrator c
        # count the number of terminals and
        # save the terminal number in Ter
        for t in range(self.nt):
            s = self.cost[t][c] - self.cost[t][self.Cassoc[t]]
            if(s < 0):
                delta[n] = s    # amount saved
                ter[n] = t      # terminal 
                n += 1
        
        if(n==0): # no terminal benefited
            return(expense)
        
        # Sort the savings, largest first and return index of sorted list
        permu = sorted(range(len(delta)), key=lambda k: delta[k]) 

        for p in permu:
            t = ter[p]
            if(delta[p] >= 0):
                break
            elif((self.weight[t]<=slack) and ((self.Cassoc[t]==self.center) 
                                           or (delta[p] + self.th_move < 0))):
                expense += delta[p]
                slack -= self.weight[t]
                
        self.logger.debug("Savings for concentrator %d is %d" % (c, expense))
        return(expense)

    def __addConc(self, c):
        delta = []
        ter = []
        for t in range(self.nt):
            s = self.cost[t][c] - self.cost[t][self.Cassoc[t]]
            if(s < 0):
                delta.append(s)    # amount saved
                ter.append(t)      # terminal 
     
        slack = self.Wlimit
        permu = sorted(range(len(delta)), key=lambda k: delta[k]) 
        #permu = [b[0] for b in sorted(enumerate(delta), key=lambda k:k[1])] 
        for p in permu:
            t = ter[p]
            if(delta[p] >= 0):
                break
            elif((self.weight[t]<=slack) and ((self.Cassoc[t]==self.center) 
                                           or (delta[p] + self.th_move < 0))):
                self.Cassoc[t] = c
                slack -= self.weight[t]

        self.logger.debug("Adding concentrator %d" % c)

# print cost list of network produced by ADD algorithm
def printCost(out, cost):   
    concList = out["conc"]
    termList = out["assoc"]
    ncenter = out["center"]
    numConc = len(concList)
    nodeAssoc = [(i,termList.count(i)-1) for i in concList]
    c = [cost[i][ncenter] for i in range(len(cost))] 
    print("Original cost =", sum(c))
    print("Central node =",ncenter)
    print("Number of concentrators =", numConc)
    print("Number of nodes per concentrators =")
    print('%4s\t%10s' % ("Conc", "Terminals"))
    for n in nodeAssoc:
        print('%4d\t%10d' % (n[0], n[1]))

    print("Total Cost =", out["cost"])


# Plot topology produced by MENTOR algorithm
def plotNetwork(out, pos, labels=[], filename="figure_add.png", 
                                        title='ADD Algorithm', show_center=False):
    numNodes = out["num"]
    center = out["center"]
    concList = out["conc"]
    edges = [(k, out["assoc"][k]) for k in range(numNodes)]
    tree = [(center, n) for n in out["conc"]]

    plt.figure(figsize=(6,6), facecolor="white")
    G=nx.path_graph(numNodes)

    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.3, 
                                              edge_color="blue")
    if show_center:
        nx.draw_networkx_edges(G, pos, edgelist=tree, width=2, 
                                              edge_color="blue", alpha=0.1)
    
    # Draw all nodes 
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="green", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[center], node_size=150, 
                                                     node_color="black")
    nx.draw_networkx_nodes(G, pos, nodelist=concList, node_size=50, 
                                                     node_color="red")

    # Draw node and edge labels       
    #elabels = {e:ch[mesh.index(e)] for e in mesh}
    #nx.draw_networkx_edge_labels(G, pos, elabels, edgelist=mesh, font_size=10,     
    #                                                        font_color="grey")
    if labels:
        nLabel = {n:labels[n] for n in concList}
        npos = {n:(pos[n][0], pos[n][1]+0.03) for n in pos}
        nx.draw_networkx_labels(G, npos, nLabel, nodelist=concList, 
                                 font_size=10, font_color="black")

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
    
