import sys
sys.path.append("/cluster/home/zhuyin/scripts/MHSXtraPy/")

from mhsxtrapy.b3d import WhichSolution
from mhsxtrapy.examples import multipole
from mhsxtrapy.field2d import Field2dData, FluxBalanceState, check_fluxbalance
from mhsxtrapy.field3d import calculate_magfield
from mhsxtrapy.plotting.vis import (
    plot_ddensity_xy,
    plot_ddensity_z,
    plot_dpressure_xy,
    plot_dpressure_z,
    plot_magnetogram_2D,
    plot_magnetogram_3D,
)

import numpy as np
import matplotlib.pyplot as plt
import sunpy
import sunpy.visualization
from astropy.visualization import ImageNormalize
from streamtracer import StreamTracer, VectorGrid

nx, ny, nz, nf = 960, 536, 536, 536
# xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

pixelsize_x = 0.23712652199468398
pixelsize_y = pixelsize_x
pixelsize_z = pixelsize_x

# x_arr = np.linspace(xmin, xmax, nx, dtype=np.float64)
# y_arr = np.linspace(ymin, ymax, ny, dtype=np.float64)
# z_arr = np.linspace(zmin, zmax, nz, dtype=np.float64)
x_arr = np.arange(nx) * pixelsize_x
y_arr = np.arange(ny) * pixelsize_y
z_arr = np.arange(nz) * pixelsize_z


data_bz = np.load("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/SOTSP/sotsp_bz_hgs_cea_fullfov.npy").T

data2d = Field2dData(
    nx,
    ny,
    nz,
    nf,
    pixelsize_x,
    pixelsize_y,
    pixelsize_z,
    x_arr,
    y_arr,
    z_arr,
    data_bz,
    flux_balance_state=FluxBalanceState.UNBALANCED,
)

data3d = calculate_magfield(
    data2d,
    alpha=0.004474636432624906,
    a=0.2,
    which_solution=WhichSolution.ASYMP,
    b=1.0,
    z0=2.0,
    deltaz=0.2,
)

print("finished 1")

data3d.save(path="/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/MHSXtra_results/SOTSP_test_full_v2/")

data3d = calculate_magfield(
    data2d,
    alpha=0.01,
    a=0.2,
    which_solution=WhichSolution.ASYMP,
    b=1.0,
    z0=2.0,
    deltaz=0.2,
)

data3d.save(path="/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/MHSXtra_results/SOTSP_test_full_pos_alpha/")

print("finished 2")

data3d = calculate_magfield(
    data2d,
    alpha=-0.01,
    a=0.2,
    which_solution=WhichSolution.ASYMP,
    b=1.0,
    z0=2.0,
    deltaz=0.2,
)

data3d.save(path="/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/MHSXtra_results/SOTSP_test_full_neg_alpha/")

print("finished 3")