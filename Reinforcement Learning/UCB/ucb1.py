#Upper confidence bound

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#implementing ucb
d=10
N=10000
no_of_sel= [0]*d
sums_of_rewards=[0]*d
ad_selected=[]
total_reward=0
for n in range(0,N):
    max_ucb=0
    ad=0
    for i in range(0,d):
        if (no_of_sel[i]>0):
            avg_reward=sums_of_rewards[i]/no_of_sel[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/no_of_sel[i])
            ucb= delta_i + avg_reward
        else:
            ucb=1e400
        if ucb>max_ucb:
            max_ucb=ucb
            ad=i
    ad_selected.append(ad)
    no_of_sel[ad]=no_of_sel[ad]+1
    reward=dataset.values[n,ad]
    sums_of_rewards[ad]+=reward
    total_reward+=reward
    
#Visualise
plt.hist(ad_selected)
plt.title('Histogram')
plt.xlabel("ads")
plt.ylabel('no of time each ad was selected')
plt.show()
