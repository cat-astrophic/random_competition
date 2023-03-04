# This script runs simulations for a paper on competition graphs of random networks

# Importing required modules

import numpy as np
import pandas as pd
import networkx as nx
import competition_graph as cg
from matplotlib import pyplot as plt
from matplotlib import cm

# Project directory info

direc = 'F:/random_competition/'

# Defining a function to generate random graphs

def random_graph(n,p):
    
    MAT = np.random.rand(n,n) # random values in matrix
    MAT = np.triu(MAT) # use upper triangular only because faster since symmetric 
    MAT = MAT > (1-p) # test if values exceed the threshold
    MAT = MAT + np.transpose(MAT) - 2*np.eye(n)*np.diag(MAT) # symmetrize and no loops
    
    return MAT

# Defining a function to generate random digraphs

def random_digraph(n,p):
    
    MAT = np.random.rand(n,n) # random values in matrix
    MAT = MAT > (1-p) # test if values exceed the threshold
    MAT = MAT - np.eye(n)*np.diag(MAT) # no loops
    
    return MAT

# Defining a function to estimate convergence in the mean degree curve for the competition graph

def converging(mean_vals, n, epsilon):
    
    checks = [n - 1 - m for m in mean_vals]
    checks = [int(c < epsilon) for c in checks]
    
    try:
        
        cp = checks.index(1) / 100
        
    except:
        
        cp = 1
    
    return cp

# Defining a function to predict mean degree

def mean_prediction(n,p):
    
    mp = (n-1) * (1 - (1-(p**2))**(n-2))
    
    return mp

# Defining a function to predict converging

# Initializing data storage

cps = []
n_data = []
epsilons = []
pred_data = []

# Main loop

domain = [5*x for x in range(1,21)]

for n in domain:
    
    nodes = []
    degrees = []
    probabilities = []
    
    for i in range(1,101): # iterate through values of p
        
        p = i/100 # define p
        
        for num in range(100): # simulations for the tuple (n,p)
            
            print('n = ' + str(n) + ' :: p = ' + str(p) + ' :: iteration = ' + str(num+1)) # visualize progress
            M = random_digraph(n,p)
            A = cg.competition_graph(M)
            degrees.append(list(sum(A)))
            nodes.append(n)
            probabilities.append(p)
            
        pred_data.append(mean_prediction(n,p))
    
    # Make plots
    
    degs = [x for d in degrees for x in d]
    probs_x = [[p]*n for p in probabilities]
    probs = [p for prob in probs_x for p in prob]    

    plot_df1 = pd.concat([pd.Series(degs, name = 'Degree'), pd.Series(probs, name = 'p')], axis = 1)
    ps = [i/100 for i in range(1,101)]
    mean_degs = [plot_df1[plot_df1.p == val].Degree.mean() for val in ps]
        
    plt.figure(figsize = (8,5))
    plt.scatter(probs, degs, s = 0.1)
    plt.plot(ps, mean_degs, color = 'red', label = 'Simulation')
    plt.plot(ps, pred_data[100*domain.index(n):100*(domain.index(n)+1)], color = 'black', linestyle = '--', label = 'Theory')
    plt.title('Mean vertex degrees in the competition graph of G(n,p) for n = ' + str(n), fontsize = 12, fontweight = 40, color = 'black')
    plt.xlabel('p')
    plt.ylabel('Degree')
    plt.legend(loc = 'lower right')
    plt.ylim(0, n)
    plt.margins(.0666,.0666)
    plt.savefig(direc + 'figures/combined_n_' + str(n) + '.png', dpi = 1000, bbox_inches = 'tight')
    plt.savefig(direc + 'figures/combined_n_' + str(n) + '.eps', dpi = 1000, bbox_inches = 'tight')
    plt.show()
    
    # Estimating convergence to dense competition graphs as a function of p
    
    epsilon_domain = [i/100 for i in range(11)]
    
    for epsilon in epsilon_domain:
        
        cps.append(converging(mean_degs, n, epsilon))
        n_data.append(n)
        epsilons.append(epsilon)

# Creating a dataframe

n_data = pd.Series(n_data, name = 'n')
cps = pd.Series(cps, name = 'p')
epsilons = pd.Series(epsilons, name = 'epsilon')
cp_df = pd.concat([n_data, cps, epsilons], axis = 1)

# Plotting the estiamted convergence threshold proababilities over n and epsilon

fig, ax = plt.subplots(subplot_kw = {'projection': '3d'})
surf = ax.plot_trisurf(cp_df.n, cp_df.epsilon, cp_df.p, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
plt.title('Probabilities at which the competition graph\n of a random network becomes (1-$ε$)-dense', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('n')
plt.ylabel('$ε$')
ax.set_zlabel('p')
#fig.colorbar(surf, shrink = 0.4, aspect = 10)
ax.view_init(30, 30)
plt.margins(.0666,.0666)
plt.savefig(direc + 'figures/estiamted_probabilities.png', dpi = 1000, bbox_inches = 'tight')
plt.savefig(direc + 'figures/estiamted_probabilities.eps', dpi = 1000, bbox_inches = 'tight')
plt.show()

# Creating a mean predictions plot over the domain

p_axis = [x/100 for x in range(1,101)]*len(domain)
n_axis = []

for n in domain:
    
    n_axis = n_axis + [n*20]*100

pred_axis = [pred_data[i] / (n_axis[i]-1) for i in range(len(pred_data))]
fig, ax = plt.subplots(subplot_kw = {'projection': '3d'})
surf = ax.plot_trisurf(p_axis, n_axis, pred_axis, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
plt.title('Predicted mean vertex degrees in the\n competition network of G(n,p) over n and p', fontsize = 12, fontweight = 40, color = 'black')
plt.xlabel('p')
plt.ylabel('n')
ax.set_zlabel('Predicted Values')
#fig.colorbar(surf, shrink = 0.4, aspect = 10)
ax.view_init(20, 60)
plt.margins(.0666,.0666)
plt.savefig(direc + 'figures/predictions_mean_degree_surf.png', dpi = 1000, bbox_inches = 'tight')
plt.savefig(direc + 'figures/predictions_mean_degree_surf.eps', dpi = 1000, bbox_inches = 'tight')
plt.show()

# Create some cool network and competition graph figures using networkx

n = [10]*4 + [20]*4
p = [.05*i for i in range(2,6)]*2

for i in range(8):
    
    M = random_digraph(n[i], p[i])
    A = cg.competition_graph(M)
    
    N = nx.digraph(M)
    C = nx.graph(A)
    
    pos = nx.circular_layout(N)

    plt.figure()
    nodes_ec = nx.draw_networkx_nodes(N, pos, alpha = 0.5)
    edges = nx.draw_networkx_edges(N, pos, edge_color = 'lightgray', arrows = False, width = 0.05)
    plt.title('An example of the random network G('+ str(n) + ',' + str(p) + ')', fontsize = 12)
    plt.margins(.0666,.0666)
    plt.axis('off')
    plt.savefig(direc + 'figures/random_network_g_' + str(n[i]) + '_' + str(int(p*100)) + '.png', dpi = 1000, bbox_inches = 'tight')
    plt.savefig(direc + 'figures/random_network_g_' + str(n[i]) + '_' + str(int(p*100)) + '.eps', dpi = 1000, bbox_inches = 'tight')
    
    plt.figure()
    nodes_ec = nx.draw_networkx_nodes(C, pos, alpha = 0.5)
    edges = nx.draw_networkx_edges(C, pos, edge_color = 'lightgray', arrows = False, width = 0.05)
    plt.title('The corresponding competition graph for G('+ str(n) + ',' + str(p) + ')', fontsize = 12)
    plt.margins(.0666,.0666)
    plt.axis('off')
    plt.savefig(direc + 'figures/random_comp_graph_g_' + str(n[i]) + '_' + str(int(p*100)) + '.png', dpi = 1000, bbox_inches = 'tight')
    plt.savefig(direc + 'figures/random_comp_graph_g_' + str(n[i]) + '_' + str(int(p*100)) + '.eps', dpi = 1000, bbox_inches = 'tight')

