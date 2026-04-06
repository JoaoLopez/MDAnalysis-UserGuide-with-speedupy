import sys
sys.path.append('/app')
from speedupy.speedupy import maybe_deterministic
import sys
sys.path.append('/app')
from speedupy.speedupy import initialize_speedupy, deterministic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from MDAnalysis.analysis import pca, align
import warnings
import time

@maybe_deterministic
def func_1():
    """Cell 1: imports executados no topo do módulo."""
    pass

@maybe_deterministic
def func_2():
    """Cell 2: carrega o Universe."""
    u = mda.Universe(PSF, DCD)
    return u

@maybe_deterministic
def func_3(u):
    """Cell 3: alinha a trajetória in-memory."""
    aligner_obj = align.AlignTraj(u, u, select='backbone', in_memory=True)
    aligner = aligner_obj.run()
    return aligner

@deterministic
def func_4(func_globals=None):
    """Cell 4: executa a PCA (gargalo).

    O Universe é recriado aqui dentro para que o SpeeduPy não tente
    serializá-lo.
    """
    u = mda.Universe(PSF, DCD)
    pca_obj = pca.PCA(u, select='backbone', align=True, mean=None, n_components=None)
    pc = pca_obj.run()
    return pc

@maybe_deterministic
def func_5(u, pc):
    """Cell 5: imprime info dos átomos backbone e shape dos componentes."""
    backbone = u.select_atoms('backbone')
    n_bb = len(backbone)
    print(f'There are {n_bb} backbone atoms in the analysis')
    print(pc.p_components.shape)
    return backbone

@maybe_deterministic
def func_6(pc):
    """Cell 6: imprime a variância do primeiro componente principal."""
    print(f'PC1: {pc.variance[0]:.5f}')

@maybe_deterministic
def func_7(pc):
    """Cell 7: imprime a variância cumulativa dos 3 primeiros PCs."""
    for i in range(3):
        print(f'Cumulated variance: {pc.cumulated_variance[i]:.3f}')

@maybe_deterministic
def func_8(pc):
    """Cell 8: plota a variância cumulativa."""
    plt.plot(pc.cumulated_variance[:10])
    plt.xlabel('Principal component')
    plt.ylabel('Cumulative variance')
    plt.show()

@maybe_deterministic
def func_9(pc, backbone):
    """Cell 9: projeta a trajetória nos 3 primeiros PCs."""
    transformed = pc.transform(backbone, n_components=3)
    return transformed

@maybe_deterministic
def func_10(transformed, u):
    """Cell 10: cria DataFrame com os componentes e o tempo."""
    df = pd.DataFrame(transformed, columns=[f'PC{i + 1}' for i in range(3)])
    df['Time (ps)'] = df.index * u.trajectory.dt
    print(df.head())
    return df

@maybe_deterministic
def func_11(pc, transformed):
    """Cell 11: calcula as coordenadas projetadas no PC1."""
    pc1 = pc.p_components[:, 0]
    trans1 = transformed[:, 0]
    projected = np.outer(trans1, pc1) + pc.mean.flatten()
    coordinates = projected.reshape(len(trans1), -1, 3)
    return coordinates

@maybe_deterministic
def func_12(backbone, coordinates):
    """Cell 12: cria um Universe a partir das coordenadas projetadas."""
    proj1 = mda.Merge(backbone)
    proj1.load_new(coordinates, order='fac')
    return proj1

@maybe_deterministic
def func_13(transformed):
    """Cell 13: calcula o cosine content dos 3 primeiros PCs."""
    for i in range(3):
        cc = pca.cosine_content(transformed, i)
        print(f'Cosine content for PC {i + 1} = {cc:.3f}')

@initialize_speedupy
def main():
    func_1()
    u = func_2()
    func_3(u)
    print('Iniciando o cálculo PCA via SpeeduPy...')
    start = time.perf_counter()
    pc = func_4(func_globals=globals())
    end = time.perf_counter()
    print(f'Tempo de execução: {end - start:.4f}s')
    backbone = func_5(u, pc)
    func_6(pc)
    func_7(pc)
    func_8(pc)
    transformed = func_9(pc, backbone)
    df = func_10(transformed, u)
    coordinates = func_11(pc, transformed)
    func_12(backbone, coordinates)
    func_13(transformed)
if __name__ == '__main__':
    main()