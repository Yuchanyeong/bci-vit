import numpy as np
t_data = np.load('data/preprocessed_wave/subject01/T/subject01_T_class1_trial004.npy')
e_data = np.load('data/preprocessed_wave/subject01/E/subject01_E_class1_trial004.npy')
print(np.allclose(t_data, e_data))  # True면 완전히 동일한 데이터!
