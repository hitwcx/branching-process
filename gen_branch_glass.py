#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:19:11 2019

@author: changxiwang
"""
# %% import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

n_g = 3
t_array = np.zeros(shape=(n_g+1,2**(n_g+1)-1), dtype=float, order='F')
t=8
#t_init = np.random.exponential(scale=4.0, size=1000)
t_init = np.random.uniform(2,4,1000)

# %% randomly grow a full tree
def gen_tree():
    #t_init = np.random.exponential(scale=3.0, size=1000)
    np.mean(t_init)
    
    n=2**(n_g+1)-1
    
    for i in range(1,n_g+1):
        for j in range(2**i):
            t_array[i,int((n-(2**i-1))*j/2**i+j):int((n-(2**i-1))*(j+1)/2**i)+j] = t_init[int((2**i+j)//2)]

    ct_array = np.cumsum(t_array,axis=0)    
    ict_array = t - ct_array

    ict_array[ict_array <= 0] = 0
    id_nonzero = np.nonzero(ict_array[n_g,:]!=0)
    
    ict2 = np.zeros((n_g+1,len(id_nonzero[0])))
    for x in range(len(id_nonzero[0])):
        ict2[:,x] = sum([],ict_array[:,id_nonzero[0][x]])
    ict2    
        
    branch_levels = []    
    for x in range(len(id_nonzero[0])):
        branch_levels.append(len(np.unique(ict2[:,x])))

    rtn_dat = [sorted(branch_levels),sum(ict_array[n_g,:]),ict2]
    return rtn_dat

tree_sum = []
levels = []
trees = []
n_rep =1

for i in range(n_rep):
    tree_data = gen_tree()
    trees.append(tree_data[2])
    tree_sum.append(tree_data[1])
    levels.append(tree_data[0])

print(trees)
levels

# %% flatten the list of times of branches
def tree_transform(trees):
    trees_unlist=reduce(lambda x,y: x+y,trees)
    return trees_unlist
trees2=tree_transform(trees)


# from full branches (including main paths) to only branches
kep_lst=list(range(trees2.shape[1]))
for i in range(1,trees2.shape[1]-1):
    flag=[]
    for j in range(i):
        if i*2-j<trees2.shape[1]:
            flag.append(all(trees2[:,j]==trees2[:,i*2-j]))
    #print(flag)
    if any(flag) or np.all(trees2[:,i]==trees2[0,i]):
        kep_lst.remove(i)
    #if np.all(trees2[:,i]==trees2[0,i]):
        #kep_lst.remove(i)
Tree=trees2[:,kep_lst]
Tree=np.vstack([Tree,Tree[-1,:]])

print(Tree)
#print(trees2)

# %% prepare drawing data sgement lengths
T=-np.diff(Tree,axis=0)
for i in range(T.shape[1]):
    if any(T[:,i]==0):
        id_zero=np.where(T[:,i]==0)[0][0]
        T[id_zero,i]=Tree[-1,i]
    else:
        continue
T

# %% prepare drawing data angles 
theta=np.random.uniform(20,30,10)
angle=np.zeros((T.shape[0],T.shape[1]))


# %%
angle[0,:]=theta[0]
dif=np.diff(T,axis=1)

# %%
