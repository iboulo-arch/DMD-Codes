import numpy as np

import matplotlib.pyplot as plt
import pandas as pd 

path=r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\profils_dmd\f_4.npy"
data=np.load(path)
data=np.array(data)
print(data.shape)
image=data.imag
plt.imshow(image)
plt.show()
df=pd.DataFrame(data)
df.head()




