# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-02 10:18:54
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-24 11:46:32
import numpy as np
import matplotlib.pyplot as plt


L = 2000
c0 = 1400.0

# wave points
t0 = 0
x0 = 0

t1 = L/c0
x1 = L

t2 = 2 * L/c0
x2 = 0

plt.plot((x0, x1), (t0, t1), label="1st segment")
plt.plot((x1, x2), (t1, t2), label="2nd segment")

# strain gauge positions
locA = 1000
#locB = 2800

plt.axvline(locA, color="k", linestyle="--")
plt.text(locA,0,'A',rotation=90)
#plt.axvline(locB, color="k", linestyle="--")

# segment duration
#delta = 2 * (locB - locA) / c0

# plot overlap times RA, RB
#RB = (L + 1 * (L - locB)) / c0
RA = (L + 1 * (L - locA)) / c0
#plt.axhline(RB, color="gray", linestyle="--")
plt.axhline(RA, color="gray", linestyle="--")

# plot segment duration 1 at A
t1 = RA
x1 = locA
#t0 = RA - delta
x0 = locA
plt.plot((x0, x1), (t0, t1), "r")

plt.show()

