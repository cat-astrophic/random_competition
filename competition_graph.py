# This script creates competition graphs from (trade, other) networks

# Importing required modules

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# Main function produces the (k-step) competition graph of a network

# Input M is the network adjacency matrix as a numpy matrix; k is the desired power of M

def competition_graph(MAT, k=1):
    
    # First, check that k is a positive integer
    
    if (type(k) == int) and (k >= 1):
        
        error_flag = False
        pass
        
    else:
        
        print('ERROR: The parameter k must be a positive integer.')
        error_flag = True
        
    # If parameterized properly, run the algorithm
    
    if error_flag != True:
        
        # Raise M to the k
        
        MAT = MAT**k - np.diag(MAT**k)*np.eye(len(MAT))
        
        # Initialize the competition graph adjacency matrix
        
        A = np.zeros((len(MAT),len(MAT)))
        
        # Run competition graph algorithm
        
        for i in range(len(MAT)):
            
            for j in range(i+1,len(MAT)):
                
                for k in range(len(MAT)):
                    
                    if (MAT[i][k] != 0) and (MAT[j][k] != 0):
                        
                        A[i][j] = 1
                        
        A = A + np.transpose(A)
                        
        return A

# Display the competition graph

# Input A is the output of competition_graph(); title is the plot title as a string 

def cg_viewer(A, title=None):
    
    # Create a graph object with networkx
    
    G = nx.Graph(A)
    
    # Create a figure
    
    plt.figure()
    nx.draw(G)
    plt.title(title)
    

