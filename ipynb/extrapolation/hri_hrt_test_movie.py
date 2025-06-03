import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
import matplotlib.patheffects as path_effects
import matplotlib.animation as animation
import sunpy
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const
from astropy.io import fits, ascii
from astropy.time import Time
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
from streamtracer import StreamTracer, VectorGrid
from extrapolater import PotentialField
from helpers import from_local

import h5py 
import dask.array as da 
from ndcube import NDCube


if __name__ == "__main__":

    file_hri_pr_noproj_dset = h5py.File("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/HRIEUV_noproj_pr.hdf5")
    hri_pr_noproj_array = file_hri_pr_noproj_dset["hrieuv_noproj_img"][:]
    hrieuv_no_proj_extent = np.load("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/hrieuv_no_proj_extent.npy")
    hrieuv_nocrop_noproj_wcs = WCS(fits.getheader("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/hri_nocrop_noproj_wcs.fits",
                                            ignore_missing_simple=True))
    hrieuv_noproj_wcs = hrieuv_nocrop_noproj_wcs[hrieuv_no_proj_extent[1]:hrieuv_no_proj_extent[3] + 1,
                                    hrieuv_no_proj_extent[0]:hrieuv_no_proj_extent[2] + 1]

    hrt_potential_field_array = np.load("/cluster/home/zhuyin/work/extrapolation/phihrt_20221024_1915/phihrt_potential_extrapolated_20221024_1915.npy")

    hrt_map = sunpy.map.Map("/cluster/home/zhuyin/Solar/extrapolate_schmidt64/data/phi_los_map_shifted_for_pore.fits")
    hrt_map = hrt_map.submap([768+16,512-64]*u.pix, top_right=[768+16+384, 512-64+384]*u.pix)
    hrt_map.meta["bunit"] = "Gauss"

    hrt_potential_ex = PotentialField(magnetogram=hrt_map, width_z=70e5*128*u.cm, shape_z=128*u.pixel)
    hrt_potential_bottom_boundary = hrt_potential_ex.project_boundary(hrt_potential_ex.range.x, hrt_potential_ex.range.y).value

    hrt_potential_field_grid_spacing = np.array([hrt_potential_ex.delta.x.to_value(u.Mm/u.pix),
                                hrt_potential_ex.delta.y.to_value(u.Mm/u.pix),
                                hrt_potential_ex.delta.z.to_value(u.Mm/u.pix)])

    hrt_potential_field_grid = VectorGrid(hrt_potential_field_array, hrt_potential_field_grid_spacing,
                                            origin_coord=[hrt_potential_ex.range.x[0].to_value(u.Mm),
                                                        hrt_potential_ex.range.y[0].to_value(u.Mm),
                                                        hrt_potential_ex.range.z[0].to_value(u.Mm)])

    seeds = np.array([[ii, jj, 0] for ii in np.linspace(0, 384, 48)*hrt_potential_ex.delta.x.to_value(u.Mm/u.pix) + hrt_potential_ex.range.x[0].to_value(u.Mm) \
                    for jj in np.linspace(0, 384, 48)*hrt_potential_ex.delta.y.to_value(u.Mm/u.pix) + hrt_potential_ex.range.y[0].to_value(u.Mm)])

    nsteps = 10000
    step_size = 0.1
    tracer = StreamTracer(nsteps, step_size)
    tracer.trace(seeds, hrt_potential_field_grid)

    fline_heeq = []
    for fline in tracer.xs[:]:
        fline_heeq.append(from_local(fline[:,0]*u.Mm, fline[:,1]*u.Mm, fline[:,2]*u.Mm, hrt_potential_ex.magnetogram.center))

    fig = plt.figure(figsize=(12,6), layout="constrained")

    ax1 = fig.add_subplot(121, projection=hrieuv_noproj_wcs)
    im1 = ax1.imshow(hri_pr_noproj_array[0,:,:], origin="lower", cmap="sdoaia171")

    ax_lim = ax1.axis()

    with propagate_with_solar_surface():
        for fline in fline_heeq[:]:
            # ax.plot_coord(fline[0], color="red", alpha=0.3, lw=2, marker="o")
            ax1.plot_coord(fline, color="cyan", alpha=0.3, lw=1)

    ax1.axis(ax_lim)

    ax2 = fig.add_subplot(122, projection=hrieuv_noproj_wcs)
    im2 = ax2.imshow(hri_pr_noproj_array[0,:,:], origin="lower", cmap="sdoaia171")

    def update_fig(ii, im1,im2, hri_pr_noproj_array):
        im1.set_data(hri_pr_noproj_array[ii,:,:])
        im2.set_data(hri_pr_noproj_array[ii,:,:])
    
    anim = animation.FuncAnimation(fig, update_fig, frames=np.arange(hri_pr_noproj_array.shape[0]),
    fargs=(im1,im2, hri_pr_noproj_array))

    anim.save("/cluster/home/zhuyin/work/extrapolation/movie/hrieuv_hrt_test_movie.mp4", writer="ffmpeg", fps=15)


