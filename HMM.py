import numpy as np
from random import random
from random import seed
import math

def viterbi_algorithm(observed_sequence):
    T = len(observed_sequence)
    N = p.shape[0]
    pi = [1,0,0,0]
    delta = np.zeros((T, N))
    W = np.zeros((T, N))
    for i in range(0,N):
        delta[0][i] = pi[i]*b[i][observed_sequence[0]-1]
        W[0][i] = 0
    Index = 0
    for t in range(1, T):
        for i in range(0, N):
            maximum = -math.inf - 1
            Index = 0
            for j in range(0,N):
                if maximum < delta[t-1][j]*p[j][i]:
                    maximum = delta[t-1][j]*p[j][i]
                    Index = j
            delta[t][i] = maximum*b[i][observed_sequence[t]-1]
            W[t][i] = Index
    for t in range(0,T):
        maximum_1 = -math.inf - 1 
        Index_1 = 0
        for j in range(0, N):
            if maximum_1 < delta[t][j]:
                maximum_1 = delta[t][j]
                Index_1 = j;
    q = np.zeros(T)
    q[T-1] = Index_1
    t = T-2
    while(t >= 0):  
        s = q[t+1]
        q[t] = W[t+1][int(s)]
        t = t-1
        
    for i in range(len(q)):
        q[i] = q[i] + 1
    #print(q)
    return q

# forward_algorithm algorithm is used to generate the probability of observation in the HMM
def forward_algorithm(observed_sequence):
        T = len(observed_sequence)
        N = p.shape[0]
        alpha = np.zeros((T, N))
        alpha[0] = pi*b[:,observed_sequence[0]-1]
        for t in range(0, T-1):
            for i in range(0, N):
                for j in range(0, N):
                    A = alpha[t][j]*p[j][i]*b[i][observed_sequence[t+1]-1]
                    alpha[t+1][i] = alpha[t+1][i] + A
        print(alpha)
        return alpha
    
def likelihood(observed_sequence):
        return  forward_algorithm(observed_sequence)[-1].sum()
    
def backward_algorithm(observed_sequence):
        N = p.shape[0]
        T = len(observed_sequence)
        beta = np.zeros((T,N))
        beta[-1:,:] = 1
        for t in (range(T-2,-1,-1)):
            for i in range(0,N):
                for j in range(0,N):
                    H = p[i][j]*b[j][observed_sequence[t+1]-1]*beta[t+1][j]
                    beta[t][i] = beta[t][i] + H
        return beta

#################### Generation of probabilities ######################
        
seed(200321563)

s = np.random.rand(4,4)
q = np.random.rand(4,3)
p = np.round(s/s.sum(axis=1)[:,None],4)
print(p)
b = np.round(q/q.sum(axis=1)[:,None],4)
print(b)
pi = [1,0,0,0]

t = 1
q = [1]
O = []
r = random()
#print(r)
if r <= b[0][0]:
    e = 1
elif r > b[0][0] and r <= (b[0][0] + b[0][1]):
    e = 2
else:
    e = 3     
O.append(e)  
   
for i in range(1,1000,1):
    d = random()
    state = q[i - 1]
    sum = 0
    comp = []
    for k in range(0,4,1):
            comp.append(sum + p[state-1][k])
            sum += p[state-1][k]
    if d <= comp[0]:
        c = 1
    else:
        f = 0
        for k in range(1, state-1):
            if d > comp[k-1] and d <= comp[k]:
                c = k + 1
                f = 1
                break
    if f == 0:
        c = len(comp)  
    q.append(c)
    h = random()
    if h <= b[c-1][0]:
        f = 1
    elif h > b[c-1][0] and h <= (b[c-1][0] + b[c-1][1]):
        f = 2
    else:
        f = 3
    O.append(f)
    
#print(q)
print(O)

############################# Task-2 ###################################

Observed = [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3]
probability = likelihood(Observed)
print("\nThe probability that the above sequence of observation came from HMM:")
print(probability)

B = backward_algorithm(Observed)
print(B)

############################# Task-3 ###################################

Sequence = viterbi_algorithm(Observed)
print("\n The most probable sequence for given observation:")
print(Sequence)

############################# Task-4 ####################################

observations = Observed
T = len(observations)
N = p.shape[0]
E = np.zeros((T,N,N))
Y = np.zeros((T,N))
alpha = forward_algorithm(observations)
beta = backward_algorithm(observations)
pi_new = np.zeros((N))
p_new = np.zeros((N,N))
b_new = np.zeros((N,3))
for t in range (0,T-1):
    sum = 0
    for i in range(0,N):
        for j in range(0,N):
            sum += alpha[t][j]*beta[t][j]
        for j in range(0,N):
            if sum != 0:
                E[t][i][j] = ((alpha[t][i]*p[i][j]*b[j][observations[t+1]-1]*beta[t+1][j])/sum)
        
for t in range (0,T-1):
    for i in range(0,N):
        sum = 0
        for j in range(0,N):
            sum += E[t][i][j]
        Y[t][i] = sum
    
for i in range(0,N):
    pi_new[i] = Y[0][i]
    
for i in range(0,N): 
    for j in range(0,N):
        value_1 = 0
        value_2 = 0  
        for t in range (0,T-1):
            value_2 += Y[t][i]
            value_1 += E[t][i][j]
        p_new[i][j] = value_1/value_2

for i in range(0,N):
    for k in range(0,3):
        value_1 = 0
        value_2 = 0
        for t in range(0,T-1):
            value_1 += Y[t][i]
            if observations[t]-1 == k:
                value_2 += Y[t][i]
        b_new[i][k] = value_2/value_1

print(p_new) 
print(b_new)