import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib import animation
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
from fancy_colorbar import plot_colorbar



import sys
sys.path.append("/cluster/home/zhuyin/scripts/MHSXtraPy/")

from mhsxtrapy.b3d import WhichSolution
from mhsxtrapy.examples import multipole
from mhsxtrapy.field2d import Field2dData, FluxBalanceState, check_fluxbalance
from mhsxtrapy.field3d import calculate_magfield, Field3dData
from mhsxtrapy.plotting.vis import (
    plot_ddensity_xy,
    plot_ddensity_z,
    plot_dpressure_xy,
    plot_dpressure_z,
    plot_magnetogram_2D,
    plot_magnetogram_3D,
)

def colored_line_with_alpha(x, y, c, ax, norm_color, norm_alpha, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    cmap = plt.get_cmap(default_kwargs.pop("cmap", "plasma"))
    color_rgb = cmap(norm_color(c))
    color_rgb[:,-1] = 1 - norm_alpha(c)*0.9

    lc = LineCollection(segments, colors=color_rgb, **default_kwargs)
    # lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


if __name__ == "__main__":
    file_Hbeta_pr = h5py.File("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/Hbeta_BJOLO_pr.hdf5")
    Hbeta_pr_set = file_Hbeta_pr["vbi_img"]
    Hbeta_pr_da = da.from_array(Hbeta_pr_set, chunks=(1, 4096 - 128*2, 4096 - 128*2))
    Hbeta_date_obs = Time(ascii.read("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/Hbeta_BJOLO_date_avg.txt")["DATE-AVG"])

    file_Halpha_pr = h5py.File("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/Halpha_BLZNL_pr.hdf5")
    Halpha_pr_set = file_Halpha_pr["vbi_img"]
    Halpha_pr_da = da.from_array(Halpha_pr_set, chunks=(1, 4096 - 128*2, 4096 - 128*2))
    Halpha_date_obs = Time(ascii.read("/cluster/home/zhuyin/work/dkist_solo_fibril_data/pid_1_123_aux/plot_ready/Halpha_BLZNL_date_avg.txt")["DATE-AVG"])

    dkist_vbi_target_header = fits.getheader("../../data/pid_1_123_aux/plot_ready/dkist_target_wcs_header_before_crop.fits",
                                            ignore_missing_simple=True)

    dkist_vbi_target_data = np.zeros((4096,4096))

    dkist_vbi_target_cube = NDCube(dkist_vbi_target_data,WCS(dkist_vbi_target_header, naxis=2))
    dkist_vbi_target_cube_crop = dkist_vbi_target_cube[128:-128,128:-128]
    dkist_vbi_target_cube_crop_rebin = dkist_vbi_target_cube_crop.rebin((8,8))

    data3d = Field3dData.load("../../data/pid_1_123_aux/MHSXtra_results/SOTSP_test_full/")

    nx, ny, nz, nf = 960, 540, 540, 540
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

    bx_extra = data3d.field[:,:,:,1]
    by_extra = data3d.field[:,:,:,0]
    bz_extra = data3d.field[:,:,:,2]

    bx_extra = bx_extra[ny:ny*2, nx:nx*2,:]
    by_extra = by_extra[ny:ny*2, nx:nx*2,:]
    bz_extra = bz_extra[ny:ny*2, nx:nx*2,:]

    field_array = np.array([bx_extra.transpose(1,0,2), by_extra.transpose(1,0,2), bz_extra.transpose(1,0,2)]).transpose(1,2,3,0)

    field_grid = VectorGrid(field_array, grid_coords=[x_arr, y_arr, z_arr])

    seeds = np.array([[ii, jj, 0] for ii in np.linspace(x_arr[420], x_arr[590], 24) for jj in np.linspace(y_arr[220], y_arr[380], 24)])

    nsteps = 10000
    step_size = 0.02
    tracer = StreamTracer(nsteps, step_size)
    tracer.trace(seeds, field_grid)

    sotsp_br_map = sunpy.map.Map("../../data/pid_1_123_aux/SOTSP/sotsp_bz_hgs_cea_fullfov.fits")

    fline_hgs = []

    for fline in tracer.xs[:]:
        # if fline[-1,2] < 60:
        fline_hgs_ = sotsp_br_map.wcs.pixel_to_world(fline[:,0]/pixelsize_x, fline[:,1]/pixelsize_y)
        fline_hgs_ = SkyCoord(lon=fline_hgs_.lon, lat=fline_hgs_.lat, radius=fline[:,2]*u.Mm + 695700*u.km,
                            frame=fline_hgs_.frame)
        
        fline_hgs.append(fline_hgs_)

    fline_vbi_pixel = []

    for fline in fline_hgs[:]:
        with propagate_with_solar_surface():
            fline_pixel_x, fline_pixel_y = dkist_vbi_target_cube_crop.wcs.world_to_pixel(fline)
        fline_vbi_pixel.append([fline_pixel_x, fline_pixel_y, fline.radius.to_value(u.Mm) - 695.7])

    del data3d, bx_extra, by_extra, bz_extra, field_array, field_grid, seeds, tracer, fline_hgs, fline_hgs_

    with plt.style.context('default'):
        fig = plt.figure(figsize=(9,8))
        
        halpha_index = 0
        hbeta_index = np.argmin(np.abs(Hbeta_date_obs - Halpha_date_obs[halpha_index]))

        ax1 = fig.add_subplot(221, projection=dkist_vbi_target_cube_crop.wcs)
        im1 = ax1.imshow(Hbeta_pr_da[hbeta_index,:,:], origin="lower", cmap="Greys_r",
                norm=ImageNormalize(vmin=0,vmax=1),
                interpolation="none")

        ax2 = fig.add_subplot(222, projection=dkist_vbi_target_cube_crop.wcs)
        im2 = ax2.imshow(Hbeta_pr_da[hbeta_index,:,:], origin="lower", cmap="Greys_r",
                norm=ImageNormalize(vmin=0,vmax=1),
                interpolation="none")

        

        ax3 = fig.add_subplot(223, projection=dkist_vbi_target_cube_crop.wcs)
        im3 = ax3.imshow(Halpha_pr_da[halpha_index,:,:], origin="lower", cmap="Greys_r",
                norm=ImageNormalize(vmin=0,vmax=0.8,
                stretch=AsinhStretch(0.8)),
                interpolation="none")

        ax4 = fig.add_subplot(224, projection=dkist_vbi_target_cube_crop.wcs)
        im4 = ax4.imshow(Halpha_pr_da[halpha_index,:,:], origin="lower", cmap="Greys_r",
                norm=ImageNormalize(vmin=0,vmax=0.8,
                stretch=AsinhStretch(0.8)),
                interpolation="none")

        for ax_ in (ax2,ax4):
            ax_lim = ax_.axis()
        
            for fline in fline_vbi_pixel[:]:
                    colored_line_with_alpha(fline[0], fline[1], fline[2], ax_, cmap="summer",
                    norm_color=ImageNormalize(vmin=0,vmax=2, clip=True),
                    norm_alpha=ImageNormalize(vmin=0,vmax=5, clip=True),
                    lw=1.5)

            ax_.axis(ax_lim)

        for ax_ in (ax1,ax2,ax4):
            ax_.coords[0].set_ticklabel_visible(False)
            ax_.coords[1].set_ticklabel_visible(False)
            ax_.coords[0].set_axislabel("")
            ax_.coords[1].set_axislabel("")

        ax3.coords[0].set_axislabel("Helioprojective Longitude (Solar-X)")
        ax3.coords[1].set_axislabel("Helioprojective Latitude (Solar-Y)")

        for ax_ in (ax1,ax2,ax3,ax4):
            ax_.grid(color='white', ls=':', lw=0.8, alpha=0.6)

        line_colormappable = ScalarMappable(norm=ImageNormalize(vmin=0,vmax=2, clip=True), cmap="summer")

        clb, clb_ax = plot_colorbar(line_colormappable, ax2, bbox_to_anchor=(1.035, 0.1, 0.06, 0.9))
        clb_ax.set_ylabel("Altitude (Mm)", fontsize=10)


        ax1.text(0.03,0.03, r"VBI H$\beta$",
                transform=ax1.transAxes, fontsize=10, ha="left", va="bottom", color="white",
                weight="bold", path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
        ax3.text(0.03,0.03, r"VBI H$\alpha$",
                transform=ax3.transAxes, fontsize=10, ha="left", va="bottom", color="white",
                weight="bold", path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
        
        fig.tight_layout()

        def update_fig(ii, ims, Hbeta_pr_da, Halpha_pr_da,
                       Hbeta_date_obs, Halpha_date_obs,
                       ):
            
            halpha_index = ii
            hbeta_index = np.argmin(np.abs(Hbeta_date_obs - Halpha_date_obs[halpha_index]))

            ims[0].set_data(Hbeta_pr_da[hbeta_index,:,:])
            ims[1].set_data(Hbeta_pr_da[hbeta_index,:,:])
            ims[2].set_data(Halpha_pr_da[halpha_index,:,:])
            ims[3].set_data(Halpha_pr_da[halpha_index,:,:])

        anim = animation.FuncAnimation(fig, update_fig, frames=range(len(Halpha_date_obs)), # range(0,50)
                                fargs=((im1, im2, im3, im4),
                                       Hbeta_pr_da, Halpha_pr_da,
                                       Hbeta_date_obs, Halpha_date_obs),
                                       blit=False)
    
        anim.save("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/figs/test_movie/vbi_sotsp_mhs_test.mp4",
            fps=30, dpi=150,
            writer='ffmpeg', 
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p'])

            

