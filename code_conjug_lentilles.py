import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1. Résolution symbolique de l'équation
z, y = sp.symbols('z y', real=True)
f_x, f = sp.Integer(300), sp.Integer(150)

y = (-f-f_x-(z*f_x)/(f_x - z))*f/(-f_x-(z*f_x)/(f_x - z))

# 2. Fonction numpy vectorisée
y_func = sp.lambdify(z, y, modules='numpy')

# 3. Tracé
z_vals = np.linspace(-1000, 1000, 4000)
y_vals = y_func(z_vals)
y_vals = np.where(np.abs(y_vals) > 5000, np.nan, y_vals)  # masque les explosions

plt.plot(z_vals, y_vals)
plt.xlabel('z'); plt.ylabel('y')
plt.grid(alpha=0.3)
plt.show()