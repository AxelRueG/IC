from os.path import abspath
from ejercicios import guia1 as g1

if __name__ == '__main__':
    # g1.ejer_1(abspath('./data/gtp1/OR_trn.csv'), 10, 0.3)
    g1.ejer_3(abspath("./data/gtp1/XOR_trn.csv"), 500, 0.2)