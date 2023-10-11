import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath


arch_name_trn = abspath('data/Guia6/leukemia_train.csv')

# --- Entrenamiento ------------------------------------------------------------------
data = np.genfromtxt(arch_name_trn, delimiter=',')

print(data.shape)

plt.stem(data[:,-2])
plt.show()

