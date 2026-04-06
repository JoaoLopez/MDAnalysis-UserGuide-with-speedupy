import sys
sys.path.append('/app')
from speedupy.speedupy import maybe_deterministic
import sys
sys.path.append('/app')
from speedupy.speedupy import initialize_speedupy, deterministic
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD, DCD2, GRO, XTC, PSF_NAMD_GBIS, DCD_NAMD_GBIS, PDB_small, CRD
from MDAnalysis.analysis import encore
import time

@maybe_deterministic
def func_1():
    """Cell 1: imports — executados no topo do módulo."""
    pass

@maybe_deterministic
def func_2():
    """Cell 2: carrega os Universes e define os labels."""
    u1 = mda.Universe(PSF, DCD)
    u2 = mda.Universe(PSF, DCD2)
    u3 = mda.Universe(GRO, XTC)
    u4 = mda.Universe(PSF_NAMD_GBIS, DCD_NAMD_GBIS)
    labels = ['DCD', 'DCD2', 'XTC', 'NAMD']
    return (u1, u2, u3, u4, labels)

@maybe_deterministic
def func_3(u1, u2, u3):
    """Cell 3: imprime o tamanho de cada trajetória."""
    print(len(u1.trajectory), len(u2.trajectory), len(u3.trajectory))

@deterministic
def func_4(func_globals=None):
    """Cell 4: calcula a Harmonic Ensemble Similarity (gargalo).

    Os Universes são recriados aqui dentro para que o SpeeduPy não tente
    serializá-los, apenas o parâmetro 'select' (string) é passado.
    """
    u1 = mda.Universe(PSF, DCD)
    u2 = mda.Universe(PSF, DCD2)
    u3 = mda.Universe(GRO, XTC)
    u4 = mda.Universe(PSF_NAMD_GBIS, DCD_NAMD_GBIS)
    (hes, details) = encore.hes([u1, u2, u3, u4], select='backbone', align=True, cov_estimator='shrinkage', weights='mass')
    return (hes, details)

@maybe_deterministic
def func_5(hes):
    """Cell 5: imprime a matriz HES formatada."""
    for row in hes:
        for h in row:
            print(f'{h:>10.1f}', end=' ')
        print('')

@maybe_deterministic
def func_6(details):
    """Cell 6: inspeciona o shape da média do ensemble 1."""
    print(details['ensemble1_mean'].shape)

@maybe_deterministic
def func_7(hes, labels):
    """Cell 7: plota a matriz de similaridade harmônica."""
    (fig, ax) = plt.subplots()
    im = plt.imshow(hes)
    plt.xticks(np.arange(4), labels)
    plt.yticks(np.arange(4), labels)
    plt.title('Harmonic ensemble similarity')
    cbar = fig.colorbar(im)
    plt.show()

@initialize_speedupy
def main():
    func_1()
    (u1, u2, u3, u4, labels) = func_2()
    func_3(u1, u2, u3)
    print('Iniciando o cálculo HES via SpeeduPy...')
    start = time.time()
    (hes, details) = func_4(func_globals=globals())
    end = time.time()
    print(f'Tempo de execução: {end - start:.4f}s')
    func_5(hes)
    func_6(details)
    func_7(hes, labels)
if __name__ == '__main__':
    main()