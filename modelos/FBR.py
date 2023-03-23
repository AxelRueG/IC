import numpy as np


class FBR:
    def __init__(self, architecture, trn_data_set, learn_coef=0.2):
        self.__architecture = architecture
        self.__trn_data = trn_data_set
        self.__radial_weigth = np.zeros(architecture[0],2)
        
    def k_means(self, input):
        num_samples = len(input)

        rgn = np.random.default_rng()
        it_random = rgn.permutation(np.arange(num_samples))

        