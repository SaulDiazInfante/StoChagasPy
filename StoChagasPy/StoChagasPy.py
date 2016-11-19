import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cPickle as pickle
from StoChagas import NumericsStochasticChagasDynamics
from matplotlib import rcParams
stoChagas = NumericsStochasticChagasDynamics()
stoChagas.set_dimensional_parameters()
stoChagas.set_dimensionless_parameters()
k = 5
p = 0
r = 0
stoChagas.initialize_mesh(k, p, r, 0.0, T=10000)
sol = stoChagas.deterministic_integration()
sol_stk = stoChagas.linear_steklov()
fig, axes = plt.subplots(nrows=4, ncols=2)
# plt.setp(axes.flat, xticks=[], yticks=[])
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flat
t = stoChagas.t
ax0.plot(t, sol[:, 0])
ax0.plot(t, sol_stk[:, 0])
ax0.set_title(r'$I_H$')
#
ax1.plot(t, sol[:, 1])
ax1.plot(t, sol_stk[:, 1])
ax1.set_title(r'$I_{A_1}$')
#
ax2.plot(t, sol[:, 2])
ax2.plot(t, sol_stk[:, 2])
ax2.set_title(r'$I_{V_1}$')
#
ax3.plot(t, sol[:, 3])
ax3.plot(t, sol_stk[:, 3])
ax3.set_title(r'$V_1$')
#
ax4.plot(t, sol[:, 4])
ax4.plot(t, sol_stk[:, 4])
ax4.set_title(r'$I_{A_2}$')
#
ax5.plot(t, sol[:, 5])
ax5.plot(t, sol_stk[:, 5])
ax5.set_title(r'$I_{V_2}$')
#
ax6.plot(t, sol[:, 6])
ax6.plot(t, sol_stk[:, 6])
ax6.set_title(r'$V_2$')
plt.tight_layout()
plt.show()
