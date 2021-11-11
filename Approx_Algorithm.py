


"""
Created on Thur October 24 10:03:37 2019

@author: Nova Fandina """


""""

The code below is the implementation of the approximation algorithm with provable guarantees, as appears
in our paper "Dimensionality Reduction: theoretical perspective on practical measures", NeurIps19.
  
The input is any finite metric space X (given by the distance matrix), an integer k>=3 anf an integer q>=2. 
The algorithm computes an embedding F of X into a k-dimensional Euclidean space, with lq_distortion(F)=1+O(q/k)*OPT,
where OPT is the l_q-distortion of the optimal embedding of X into k-dimensional Euclidean space.
  
The algorithm performs two steps: 

1. Computes an optimal embedding of X into high dimensional Euclidean space (i.e., without restricting the traget dimension.).
   The objective is the lq_distortion function. This step is solved by the convex programming (implemented in the cvxpy python package).
   
2. Applies the JL projection on the resulted metric from the first step, while projecting into the k dimensions.


We give here the implementation for optimizing the lq_distortion, while optimizing for the other measures is done similarly.
    
"""



import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import scipy
import math
import cvxpy as cp
from numpy import linalg as LA







"""
We first provide an auxiliary function we need for our implementation.

"""



def add_first_zero_row(matrix):
    [size, dim]=matrix.shape
    #print("The size of the materix is:", size, dim)
    new_matrix=np.zeros((size+1, dim))
    #print("the size of the new matrix is:", new_matrix)
    for i in range(1,size+1):
        #print("The index of the row i:",i)
        for j in range(dim):
            #print("The index of the column is:",j)
            new_matrix[i,j]=matrix[i-1,j]
    return(new_matrix);






#=============================================================================================
# Transform: the JL embedding into k dimensions.
#Input: high dimensional vectors. Output: low dimensional vectors.

def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    result=transformer.fit_transform(space)
    return(result)



#==================================================================================================================
#Our Approximaton algorithm


#Input: a metric space on n points, given as the distance matrix. Outputs: the vectors of the embedding into k dim.
#The embedding is a good approx to the best possible. Outputs the vectors of the embedded space.



def Approx_Algo(input_dists, new_dim, q):
    [rows, cols]=input_dists.shape
    
    #Step1: convex optimization, using the implementation package cvxpy.

    #Normalize the input metric by the largest dsiatnce (this should not change the optimal embedding,
    #but this helps to speed up the computations and to make them more precise).
    max_dist=np.amax(input_dists)
    div_input_dists=np.divide(input_dists, max_dist)

    G=cp.Variable((rows-1,rows-1), PSD=True)

    #Z i sthe matrix of the new dists, squared
    Z=cp.Variable((rows,rows),symmetric=True)

    #E is the matrix of expansions, squared
    E=cp.Variable((rows, rows), symmetric=True)

    #C is the matrix of contructions, squared
    C=cp.Variable((rows, cols), symmetric=True)
    C=cp.inv_pos(E)

    #M is the matrix of distortions, squared
    M=cp.Variable((rows, cols),symmetric=True)
    M=cp.maximum(E, C)


    one=cp.Parameter()
    one.value=1

    #the constraints describe the convex boundary set
    constraints=[]
    for j in range(1,rows):
        constraints=constraints+[Z[0,j]==G[j-1,j-1]]

    for i in range(1, rows):
        for j in range(i+1,rows):
            constraints=constraints+[Z[i,j]==G[i-1,i-1]+G[j-1,j-1]-2*G[i-1,j-1]]

    for i in range(rows):
        for j in range(i+1,rows):
            constraints=constraints+[Z[i,j]>=0]

    for i in range(rows):
            constraints=constraints+[E[i,i]==one]


    for i in range (rows):
        for j in range (i+1, rows):
            constraints=constraints+[Z[i,j]==E[i,j]*(div_input_dists[i,j]**2)]


    #The optimization objective function is l_q-distrotion.
    prob=cp.Problem(cp.Minimize(cp.pnorm(M, p=q/2)),constraints)
    prob.solve()


    #Recovering the resultimg vectors of the embedding from the distances, by computing eigenvalue decomposition of G.
    eig_vals, eig_vectors=np.linalg.eigh(G.value)
    num_eigs=len(eig_vals)
    D_matrix=np.zeros((num_eigs, num_eigs))
    for i in range(num_eigs):
         D_matrix[i,i]=math.sqrt(abs(eig_vals[i]))

    #The rows of U should be the orthonormal basis of the eig_vectors.
    U_matrix=np.transpose(eig_vectors)
    the_vectors=np.matmul(D_matrix, U_matrix)

    #The original vectors are the cols of the above matrix.
    recov_vecs=np.transpose(the_vectors)

    #The assumption is that the first vector is mapped to 0 vector. So we bring it back.
    vectors=add_first_zero_row(recov_vecs)


    #Note:  We could use the Cholesky decomposition of python,
    #but there are floating point issues, so we implemented our own decomposition.

    #Step 2: embed the high dimimensional vectors into vectors of dimension new_dim, with the JL projection.
    #Output is the set of vectors in low dimension.
 

    low_dim_space=JL_transf(vectors, new_dim)

    #Bring the normalization factor back.
    real_low_dim_space=low_dim_space*max_dist
    return(real_low_dim_space);



    
