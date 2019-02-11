import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 22})


fig, (ax1) = plt.subplots()
t = np.array([5, 10, 15, 20, 25, 30])
y1_mod1 = np.array([0.32, 0.34, 0.35 , 0.36 , 0.37, 0.38])

y1_mod2 = np.array([0.42, 0.43, 0.45, 0.47, 0.49, 0.54 ])


y2_mod1 = np.array([3,  8 , 12 , 16 , 18, 21])
y2_mod2 = np.array([2 , 7, 9, 10, 12 , 13 ])
#ax1.plot(t[0],y1_mod1[0], linestyle='--', marker='s', color= 'k', label='SocialGAN')
#ax1.plot(t[0],y1_mod2[0], linestyle=':', marker='^', color= 'k', label='SafeGAN')

ax1.plot(t, y1_mod1, linestyle='--', marker='s', color= 'k',  linewidth =1)
ax1.plot(t, y1_mod2, linestyle=':', marker='^', color= 'k',  linewidth =1)
ax1.axis('tight')
ax1.set_ylabel('Mean Euclidean Distance (m)')
ax1.set_xlabel('Num Trajectories (N)')


#ax2.plot(t, y2_mod1, linestyle='--', marker='s', color= 'k', linewidth =1, label='SocialGAN')
#ax2.plot(t, y2_mod2, linestyle=':', marker='^', color= 'k',  linewidth =1, label='SafeGAN')
#ax2.set_ylabel('Num Collisions (N)')
#ax2.set_xlabel('Num Trajectories (N)')
#ax2.legend()

plt.tight_layout()


plt.show()
