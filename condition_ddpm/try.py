import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch

s_curve, _ = make_s_curve(10**4,noise=0.1)
s_curve = s_curve[:,[0,2]]/10.0

plt.scatter(s_curve[:1000,0], s_curve[:1000,1])
plt.show()
plt.savefig('s2.png')
plt.close()