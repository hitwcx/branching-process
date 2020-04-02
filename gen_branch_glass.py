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

n_g = 6
t_array = np.zeros(shape=(n_g+1,2**(n_g+1)-1), dtype=float, order='F')
t=9
#t_init = np.random.exponential(scale=4.0, size=1000)
t_init = np.random.uniform(2,3,1000)

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

print(np.round(trees,1))
levels

# %% flatten the list of times of branches
def tree_transform(trees):
    trees_unlist=reduce(lambda x,y: x+y,trees)
    return trees_unlist
trees2=tree_transform(trees)
print(np.round(trees2,2))

# %% from full branches (including main paths) to only branches
ext_lst=list(range(trees2.shape[1]))
rem_lst=[]
peels=[]

while len(ext_lst)!=1:
    ext_copy=ext_lst.copy()
    for i in range(1,len(ext_lst)-1):
        if all(trees2[:,ext_lst[i-1]]==trees2[:,ext_lst[i+1]]):
            rem_lst.append(ext_lst[i])
            peels.extend([ext_lst[i-1],ext_lst[i+1]])
            ext_copy.remove(ext_lst[i-1])
            ext_copy.remove(ext_lst[i+1])
    ext_lst=ext_copy.copy()
print(rem_lst)

set_ext=set(range(trees2.shape[1]))
set_rem=set(rem_lst)
kept_lst=list(set_ext.difference(set_rem))
Tree=trees2[:,kept_lst]

print(np.round(Tree,1))
#print(trees2)

# %% prepare drawing data sgement lengths
T=-np.diff(Tree,axis=0)
for i in range(T.shape[1]):
    if any(T[:,i]==0):
        id_zero=np.where(T[:,i]==0)[0][0]
        T[id_zero,i]=Tree[-1,i]
    else:
        continue
print(np.round(T,2))

# %% prepare drawing data angles 
theta=list(np.random.uniform(10,15,1000))
theta0=list(np.random.uniform(10,15,1000))
theta1=[]
for ele in theta0:
    theta1.extend([-ele/2,ele/2])

T_f=T[~(T==0).all(1)]
T_f=np.vstack((T_f,np.zeros((1,T_f.shape[1]))))
angle=np.zeros((T_f.shape[0],T_f.shape[1]))

# set up the first branch (shared by all paths)
ang_ini=theta.pop()
angle[0,:]=ang_ini
size=T_f.shape[1]
 
# %%

for i in range(1,T_f.shape[0]):
    row_unique=list(set(T_f[i,:]))
    try:
        row_unique.remove(0)
    except ValueError:
        pass  # do nothing!

    for unique_value in row_unique:
        if  all((T_f[i+1,(T_f[i,:]==unique_value)]==0)):
            the_pir=theta.pop()
            angle[i,T_f[i,:]==unique_value]=np.array([the_pir/2,-the_pir/2])+np.array(angle[i-1,T_f[i,:]==unique_value])
        else:
            angle[i,T_f[i,:]==unique_value]=theta1.pop()+np.array(angle[i-1,T_f[i,:]==unique_value])
print(np.round(angle,2))
print(np.round(T_f,2))



# %%
