import numpy as np
from random import random
from random import seed
import math
import matplotlib.pyplot as plt

def forward_algorithm(p,observed_sequence):
        T = len(observed_sequence)
        N = p.shape[0]
        alpha = np.zeros((T, N))
        alpha[0] = pi*b[:,observed_sequence[0]-1]
        for t in range(0, T-1):
            for i in range(0, N):
                for j in range(0, N):
                    alpha[t+1][i] += alpha[t][j]*p[j][i]*b[i][observed_sequence[t+1]-1] 
        #print(alpha)
        return alpha
    
def likelihood(p,observed_sequence):
        return  forward_algorithm(p,observed_sequence)[-1].sum()

seed(200321563)

obj = 3
count = 802
O_all = []
probability = []
AIC = []
BIC = []
L = []
states = []
for state in range(2,count,100):
    s = np.random.rand(state,state)
    states.append(state)
    #print(s)
    p = s/s.sum(axis=1)[:,None]
    q = np.random.rand(state,obj)
    b = q/q.sum(axis=1)[:,None]
    #print(p)
    #print(b)
    pi=[]
    pi.append(1)
    
    c = state-1
    
    while(c>0):
        pi.append(0)
        c-=1
    #print(pi)
    #pi = [1,0,0,0]
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
    #print(O)
    x = obj*obj + state*obj - 1 
    #print(x)
    v = x * 10
    #print(v)
    for i in range(1,v,1):
        d = random()
        state_1 = q[i - 1]
        sum = 0
        comp = []
        for k in range(0,state,1):
            comp.append(sum + p[state_1 -1][k])
            sum += p[state_1 -1][k]
        
        #print(comp)
        
        if d <= comp[0]:
            c = 1
        else:
            f = 0
            for k in range(1, state-1):
                if d > comp[k-1] and d <= comp[k]:
                    c = k + 1
                    f = 1
                    break
        f = 0
        if f == 0:
            c = len(comp)          
        #print(c)
        q.append(c)
        h = random()
        #print(h)
        if h <= b[c-1][0]:
            f = 1
        elif h > b[c-1][0] and h <= (b[c-1][0] + b[c-1][1]):
            f = 2
        else:
            f = 3
        O.append(f)
    #print(O)
    L = O[:12]
    #print(L)
    O_all.append(L)
    for O_o in O_all :
        #print(O_o)
        prob = likelihood(p,O_o)
    probability.append(prob)
    
    #print(x)
    IC_1 = -2*math.log(prob) + 2*x
    #print(IC_1)
    AIC.append(IC_1)
    IC_2 = -2*math.log(prob) + x*math.log(len(O_all))
    #print(IC_2)
    BIC.append(IC_2)

#print(O_all)        
print(probability)  
print(AIC)
print(BIC)

plt.title('Graph')
plt.plot(states,probability,label = 'Likelihood')
plt.plot(states,AIC,label = 'AIC')
plt.plot(states,BIC,label = 'BIC')

plt.legend()