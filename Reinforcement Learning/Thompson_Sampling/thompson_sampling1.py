#Thomspon Sampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#Implement TS
d=10
N=10000
r1= [0]*d
r0=[0]*d
ad_selected=[]
total_reward=0
for n in range(0,N):
    max_random=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(r1[i]+1,r0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ad_selected.append[ad]
    reward=dataset.values(n,ad)
    if reward==1:
        r1[ad]+=1
    else:
        r0[ad]+=1
    total_reward+=1