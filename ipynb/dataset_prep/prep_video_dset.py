import numpy as np
import sunpy
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import dkist 
from sjireader import read_iris_sji
from astropy.visualization import (AsinhStretch, ImageNormalize)
import os 
from copy import deepcopy
from glob import glob 
from watroo import wow
from ndcube import NDCube
import h5py
import dask.array as da
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage
from iris_prep_despike import iris_prep_despike

def job_hrieuv_wow_and_reproject(map_raw, map_wcs, target_wcs, noproj_extent):
    """
    This function takes an IRIS SJI map in the rawest format,
    applies the wow algorithm to it, and then reprojects it to the target WCS.
    It also crops the map to the region of interest.
    The function returns the reprojected map, the non-reprojected but cropped map,
    and the date of the observation.
    """
    hrieuv_map = NDCube(map_raw.data, map_wcs)
    hrieuv_map_date_ear = map_raw.meta["date_ear"]
    hrieuv_map = hrieuv_map[500:750,1050:1360]

    with propagate_with_solar_surface():
        hrieuv_map_nowow_reproj = hrieuv_map.reproject_to(target_wcs,algorithm="adaptive")
    
    hrieuv_map = NDCube(wow(hrieuv_map.data, bilateral=1, denoise_coefficients=[5,5],)[0],
                               hrieuv_map.wcs)
    
    hrieuv_map_noproj = hrieuv_map[noproj_extent[1] - 500:noproj_extent[3] + 1 - 500,
                                   noproj_extent[0] - 1050:noproj_extent[2] + 1 - 1050]
    
    with propagate_with_solar_surface():
        hrieuv_map = hrieuv_map.reproject_to(target_wcs,algorithm="adaptive")
    
    return hrieuv_map.data, hrieuv_map_noproj.data, hrieuv_map_nowow_reproj.data, hrieuv_map_date_ear

def job_irissji_wow_and_reproject(map_raw, map_wcs, target_wcs):
    """
    This function takes an IRIS SJI map in the rawest format,
    applies the wow algorithm to it, and then reprojects it to the target WCS.
    It also crops the map to the region of interest.
    The function returns the reprojected map and the date of the observation.
    """
    irissji_map = NDCube(map_raw.data, map_wcs)
    irissji_map_date = map_raw.meta["date-obs"]
    irissji_map = irissji_map[100:300,150:350]

    with propagate_with_solar_surface():
        irissji_map_nowow_map = irissji_map.reproject_to(target_wcs,algorithm="adaptive")
    
    irissji_map = NDCube(wow(irissji_map.data, bilateral=1, denoise_coefficients=[5,5],)[0],
                               irissji_map.wcs)
    
    with propagate_with_solar_surface():
        irissji_map = irissji_map.reproject_to(target_wcs,algorithm="adaptive")
    
    return (irissji_map.data, irissji_map_nowow_map.data, irissji_map_date)

def job_aia_wow_and_reproject(map_raw, map_wcs, target_wcs):
    """
    This function takes an AIA map in the rawest format,
    applies the wow algorithm to it, and then reprojects it to the target WCS.
    It also crops the map to the region of interest.
    The function returns the reprojected map and the date of the observation.
    """
    aia_map = NDCube(map_raw.data, map_wcs)
    aia_map_date = map_raw.meta["date-obs"]
    aia_map = aia_map[250:400,280:430]
    
    aia_map = NDCube(wow(aia_map.data, bilateral=1, denoise_coefficients=[5,5],)[0],
                               aia_map.wcs)
    
    with propagate_with_solar_surface():
        aia_map = aia_map.reproject_to(target_wcs,algorithm="adaptive")
    
    return (aia_map.data, aia_map_date)

def job_reproj_vbir_to_vbib(vbir_data, vbir_wcs, target_wcs, shift):
    if shift is not None:
        vbir_map = NDCube(ndimage.shift(vbir_data, shift), vbir_wcs)
    else:
        vbir_map = NDCube(vbir_data, vbir_wcs)

    vbir_map = vbir_map.reproject_to(target_wcs,algorithm="adaptive")
    return vbir_map.data

def job_shift_img(img, shift):
    if shift is not None:
        return ndimage.shift(img, shift)
    else:
        return img


if __name__ == "__main__":

    todo_list = {"Gband":False,
                "CaIIK":False,
                "Hbeta":False,
                "Halpha":False,
                "TiO":False,
                "HRIEUV":False,
                "IRISSJI":True,
                "AIA":False,
                "VBI_DATE_OBS":False,
                }

    max_workers = 20

    vbi_gband_xshift, vbi_gband_yshift = 4.40*u.arcsec, 1.46*u.arcsec

    # fake header for reprojection, we don't have to reproject to a resolution similar to DKIST
    # for AIA and IRIS as the angle of separation is very small, we let them have the same rsun_ref

    dkist_vbi_target_header = fits.getheader("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/AEZDV/VBI_2022_10_24T18_59_10_640_00430500_I_AEZDV_L1.fits", ext=1)
    dkist_vbi_target_header["CRVAL1"] = dkist_vbi_target_header["CRVAL1"] + vbi_gband_xshift.to_value(u.arcsec)
    dkist_vbi_target_header["CRVAL2"] = dkist_vbi_target_header["CRVAL2"] + vbi_gband_yshift.to_value(u.arcsec)
    dkist_vbi_target_header["NAXIS"] = 2
    dkist_vbi_target_header.remove("NAXIS3")
    dkist_vbi_target_header.tofile("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/dkist_target_wcs_header_before_crop.fits", overwrite=True)

    dkist_vbi_target_data = np.zeros((4096,4096))

    dkist_vbi_target_cube = NDCube(dkist_vbi_target_data,WCS(dkist_vbi_target_header, naxis=2))
    dkist_vbi_target_cube_crop = dkist_vbi_target_cube[128:-128,128:-128]
    dkist_vbi_target_cube_crop_rebin = dkist_vbi_target_cube_crop.rebin((8,8))

    dkist_vbi_target_header["rsun_ref"] = 695700000 + 4.7e6 # increase some height to have a better result
    dkist_vbi_target_cube_47 = NDCube(dkist_vbi_target_data,WCS(dkist_vbi_target_header, naxis=2))
    dkist_vbi_target_cube_47_crop = dkist_vbi_target_cube_47[128:-128,128:-128]
    dkist_vbi_target_cube_47_crop_rebin = dkist_vbi_target_cube_47_crop.rebin((8,8))

    # load destretched VBI-R TiO dataset and shift WCS
    # currently we cannot directly modify the gWCS of the dkist dataset
    # so we read the FITS header and change them

    file_TiO_destretched = h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BNRPZ_aligned/BNRPZ_aligned_all.h5")
    TiO_dset = file_TiO_destretched["vbi_img"]
    TiO_da = da.from_array(TiO_dset, chunks=(1, 4096, 4096))

    TiO_dset_raw = dkist.load_dataset("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BNRPZ")

    TiO_ds_raw_header_0 = fits.getheader("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BNRPZ/VBI_2022_10_24T18_58_12_753_00705800_I_BNRPZ_L1.fits", ext=1)
    TiO_ds_raw_header_0["CDELT1"] = TiO_ds_raw_header_0["CDELT1"]*1.1075
    TiO_ds_raw_header_0["CDELT2"] = TiO_ds_raw_header_0["CDELT2"]*1.1075
    TiO_ds_raw_header_0["CRVAL1"] = TiO_ds_raw_header_0["CRVAL1"] - 13 + \
        (-140.1 - 15.2)*TiO_ds_raw_header_0["CDELT1"]*TiO_ds_raw_header_0["PC1_1"] + \
        (-125.1)*TiO_ds_raw_header_0["CDELT2"]*TiO_ds_raw_header_0["PC1_2"] + 4.4
    TiO_ds_raw_header_0["CRVAL2"] = TiO_ds_raw_header_0["CRVAL2"] - 15 + \
        (-140.1 - 15.2)*TiO_ds_raw_header_0["CDELT1"]*TiO_ds_raw_header_0["PC2_1"] + \
        (-125.1)*TiO_ds_raw_header_0["CDELT2"]*TiO_ds_raw_header_0["PC2_2"] + 1.46 # crude estimation
    TiO_ds_raw_header_0.remove("NAXIS3")

    TiO_ds_wcs = WCS(TiO_ds_raw_header_0, naxis=2)

    # load destretched VBI-R Halpha dataset
    # we will shift the image later so we don't modify the WCS

    file_Halpha_destretched = h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BLZNL_aligned/BLZNL_aligned_all.h5")
    Halpha_dset = file_Halpha_destretched["vbi_img"]
    Halpha_da = da.from_array(Halpha_dset, chunks=(1, 4096, 4096))
    Halpha_dset_raw = dkist.load_dataset("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BLZNL")

    # load destretched VBI-B G-band dataset
    # which uses the target WCS

    file_Gband_destretched = h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/AEZDV_aligned/AEZDV_aligned_all.h5")
    Gband_dset = file_Gband_destretched["vbi_img"]
    Gband_da = da.from_array(Gband_dset, chunks=(1, 4096, 4096))

    Gband_dset_raw = dkist.load_dataset("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/AEZDV/")

    # load destretched VBI-B Hbeta dataset 
    # will be shifted to match the G-band WCS 

    file_Hbeta_destretched = h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BJOLO_aligned/BJOLO_aligned_all.h5")
    Hbeta_dset = file_Hbeta_destretched["vbi_img"]
    Hbeta_da = da.from_array(Hbeta_dset, chunks=(1, 4096, 4096))

    Hbeta_dset_raw = dkist.load_dataset("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BJOLO")

    # load destretched VBI-B Ca II K dataset
    # no change to pointing because it is difficult to coalign Ca II K 

    file_CaIIK_destretched = h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BZPOW_aligned/BZPOW_aligned_all.h5")
    CaIIK_dset = file_CaIIK_destretched["vbi_img"]
    CaIIK_da = da.from_array(CaIIK_dset, chunks=(1, 4096, 4096))

    CaIIK_dset_raw = dkist.load_dataset("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123/BZPOW")

    # G-band plot-ready dataset

    if todo_list["Gband"]:
        Gband_shape = list(Gband_da.shape)
        Gband_shape[1] = Gband_shape[1] - 128*2
        Gband_shape[2] = Gband_shape[2] - 128*2
        Gband_pr_dset = np.array(Gband_shape)

        Gband_median_0 = np.nanmedian(Gband_da[0,128:-128,128:-128].compute())
        Gband_norm = ImageNormalize(vmin=np.nanpercentile(Gband_da[0,128:-128,128:-128].compute()/Gband_median_0, 1),
                                    vmax=np.nanpercentile(Gband_da[0,128:-128,128:-128].compute()/Gband_median_0, 99))

        Gband_pr_dset = Gband_norm((Gband_da[:,128:-128,128:-128]/np.nanmedian(Gband_da[:,128:-128,128:-128], axis=(1,2))[:,np.newaxis,np.newaxis]).compute())
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Gband_AEZDV_pr.hdf5","w") as hf:
            hf.create_dataset('vbi_img', data=Gband_pr_dset)
        
        del Gband_pr_dset

        print("Gband done")

    # Ca II K plot-ready dataset 

    if todo_list["CaIIK"]:
        CaIIK_median_0 = np.nanmedian(CaIIK_da[0,128:-128,128:-128].compute())
        CaIIK_norm = ImageNormalize(vmin=np.nanpercentile(CaIIK_da[0,128:-128,128:-128].compute()/CaIIK_median_0, 1),
                                    vmax=np.nanpercentile(CaIIK_da[0,128:-128,128:-128].compute()/CaIIK_median_0, 99))

        CaIIK_pr_dset = CaIIK_norm((CaIIK_da[:,128:-128,128:-128]/np.nanmedian(CaIIK_da[:,128:-128,128:-128], axis=(1,2))[:,np.newaxis,np.newaxis]).compute())
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/CaIIK_BZPOW_pr.hdf5","w") as hf:
            hf.create_dataset('vbi_img', data=CaIIK_pr_dset)
        
        del CaIIK_pr_dset

        print("CaIIK done")

    # Hbeta plot-ready dataset

    if todo_list["Hbeta"]:
        n_Hbeta_files = Hbeta_da.shape[0]

        Hbeta_median_0 = np.nanmedian(Hbeta_da[0,128:-128,128:-128].compute())
        Hbeta_norm = ImageNormalize(vmin=np.nanpercentile(Hbeta_da[0,128:-128,128:-128].compute()/Hbeta_median_0, 1),
                                    vmax=np.nanpercentile(Hbeta_da[0,128:-128,128:-128].compute()/Hbeta_median_0, 99))
        
        Hbeta_pr_dset = Hbeta_norm((Hbeta_da[:,:,:]/np.nanmedian(Hbeta_da[:,128:-128,128:-128], axis=(1,2))[:,np.newaxis,np.newaxis]).compute())

        # shift Hbeta image to match Gband

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            Hbeta_pr_dset = np.array(list(executor.map(job_shift_img, Hbeta_pr_dset, [0, 1.3]*n_Hbeta_files)))

        Hbeta_pr_dset = Hbeta_pr_dset[:,128:-128,128:-128]

        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Hbeta_BJOLO_pr.hdf5","w") as hf:
            hf.create_dataset('vbi_img', data=Hbeta_pr_dset)
        
        del Hbeta_pr_dset

        print("Hbeta done")

    # Halpha plot-ready dataset

    if todo_list["Halpha"]:

        Halpha_median_0 = np.nanmedian(Halpha_da[0,128:-128,128:-128].compute())
        Halpha_norm = ImageNormalize(vmin=np.nanpercentile(Halpha_da[0,128:-128,128:-128].compute()/Halpha_median_0, 1),
                                    vmax=np.nanpercentile(Halpha_da[0,128:-128,128:-128].compute()/Halpha_median_0, 99))

        n_Halpha_files = Halpha_da.shape[0]

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results_halpha = np.array(list(executor.map(job_reproj_vbir_to_vbib, Halpha_da.compute(), [TiO_ds_wcs]*n_Halpha_files,
                            [dkist_vbi_target_cube_crop.wcs]*n_Halpha_files, [(0,-14.3)]*n_Halpha_files)))

        Halpha_pr_dset = Halpha_norm(results_halpha/np.nanmedian(results_halpha, axis=(1,2))[:,np.newaxis,np.newaxis])

        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Halpha_BLZNL_pr.hdf5","w") as hf:
            hf.create_dataset('vbi_img', data=Halpha_pr_dset)
        
        del Halpha_pr_dset

        print("Halpha done")

    # TiO plot-ready dataset
    if todo_list["TiO"]:
        TiO_median_0 = np.nanmedian(TiO_da[0,128:-128,128:-128].compute())
        TiO_norm = ImageNormalize(vmin=np.nanpercentile(TiO_da[0,128:-128,128:-128].compute()/TiO_median_0, 1),
                                    vmax=np.nanpercentile(TiO_da[0,128:-128,128:-128].compute()/TiO_median_0, 99))

        n_TiO_files = TiO_da.shape[0]

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results_TiO = np.array(list(executor.map(job_reproj_vbir_to_vbib, TiO_da.compute(), [TiO_ds_wcs]*n_TiO_files,
                            [dkist_vbi_target_cube_crop.wcs]*n_TiO_files, [None]*n_TiO_files)))

        TiO_pr_dset = TiO_norm(results_TiO/np.nanmedian(results_TiO, axis=(1,2))[:,np.newaxis,np.newaxis])

        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/TiO_BNRPZ_pr.hdf5","w") as hf:
            hf.create_dataset('vbi_img', data=TiO_pr_dset)
        
        del TiO_pr_dset

        print("TiO done")

    # for eui we need to WOW those files and reproject 

    if todo_list["HRIEUV"]:
        eui_files = sorted(glob("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/HRI/coalign_step_boxcar/*.fits"))
        n_eui_files = len(eui_files)
        eui_map_seq_coalign = sunpy.map.Map(eui_files[:],sequence=False,memmap=True)
        
        Txshift_hri, Tyshift_hri = (1.66986 + 2.49223)*u.arcsec,(7.60204 - 2.76366 - 1.0 )*u.arcsec

        if n_eui_files < 181 + 1: # for test use
            eui_map_181 = eui_map_seq_coalign[0].shift_reference_coord(Txshift_hri,Tyshift_hri)
        else: # for the entire dataset
            eui_map_181 = eui_map_seq_coalign[181].shift_reference_coord(Txshift_hri,Tyshift_hri)

        eui_map_181.meta["rsun_ref"] = 695700000 + 4.7e6
        eui_map_wcs = eui_map_181.wcs

        eui_map_wcs.to_header().tofile("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/hri_nocrop_noproj_wcs.fits",
                                        overwrite=True)
        
        with propagate_with_solar_surface():
            hrieuv_no_proj_left_bottom_x, _ = \
            np.floor(eui_map_wcs.world_to_pixel(dkist_vbi_target_cube_47_crop.wcs.pixel_to_world(0,4096-1-128*2))).astype("int")

            _, hrieuv_no_proj_left_bottom_y = \
            np.floor(eui_map_wcs.world_to_pixel(dkist_vbi_target_cube_47_crop.wcs.pixel_to_world(0,0))).astype("int")

            hrieuv_no_proj_top_right_x, _ = \
            np.ceil(eui_map_wcs.world_to_pixel(dkist_vbi_target_cube_47_crop.wcs.pixel_to_world(4096-1-128*2,0))).astype("int")

            _, hrieuv_no_proj_top_right_y = \
            np.ceil(eui_map_wcs.world_to_pixel(dkist_vbi_target_cube_47_crop.wcs.pixel_to_world(4096-1-128*2,4096-1-128*2))).astype("int")

            hrieuv_no_proj_extent = [hrieuv_no_proj_left_bottom_x, hrieuv_no_proj_left_bottom_y,
                                    hrieuv_no_proj_top_right_x, hrieuv_no_proj_top_right_y]
        
        np.save("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/hrieuv_no_proj_extent.npy", hrieuv_no_proj_extent)


        # run job_hrieuv_wow_and_reproject and collect results using ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_hri = list(executor.map(job_hrieuv_wow_and_reproject, eui_map_seq_coalign,
            [eui_map_wcs]*n_eui_files, [dkist_vbi_target_cube_47_crop_rebin.wcs]*n_eui_files,
            [hrieuv_no_proj_extent]*n_eui_files))

        hrieuv_pr_dset = np.array([r[0] for r in results_hri])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/HRIEUV_pr.hdf5","w") as hf:
            hf.create_dataset('hrieuv_img', data=hrieuv_pr_dset)

        hrieuv_noproj_pr_dset = np.array([r[1] for r in results_hri])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/HRIEUV_noproj_pr.hdf5","w") as hf:
            hf.create_dataset('hrieuv_noproj_img', data=hrieuv_noproj_pr_dset)

        hrieuv_nowow_pr_dset = np.array([r[2] for r in results_hri])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/HRIEUV_nowow_pr.hdf5","w") as hf:
            hf.create_dataset('hrieuv_nowow_img', data=hrieuv_nowow_pr_dset)

        hrieuv_date_ear = [r[3] for r in results_hri]
        hrieuv_date_ear_table = Table()
        hrieuv_date_ear_table["date_ear"] = hrieuv_date_ear
        hrieuv_date_ear_table.write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/HRIEUV_date_ear.txt",
                                    format="ascii", overwrite=True)

        del hrieuv_pr_dset
        del hrieuv_noproj_pr_dset
        del hrieuv_nowow_pr_dset

        print("HRIEUV done")

    # IRIS SJI 1400 plot-ready dataset

    if todo_list["IRISSJI"]:

        iris_sji_1400_cube = read_iris_sji(
            "/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/IRIS/iris_l2_20221024_190447_3643101203_SJI_1400_t000.fits",
            sdo_rsun=False)
        
        n_iris_sji_1400 = len(iris_sji_1400_cube)

        iris_sji_1400_cube = sunpy.map.Map(iris_sji_1400_cube, sequence=True)
        iris_sji_1400_cube_data = iris_sji_1400_cube.data
        iris_sji_1400_cube_data_despike = iris_prep_despike(iris_sji_1400_cube_data, sigmas=4, mode="both")
        iris_sji_1400_cube_despike = []

        for ii in range(n_iris_sji_1400):
            iris_sji_1400_cube_despike.append(
                sunpy.map.Map(iris_sji_1400_cube_data_despike[:,:,ii], iris_sji_1400_cube[ii].meta)
            )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_irissji_1400 = list(executor.map(job_irissji_wow_and_reproject, iris_sji_1400_cube_despike,
            [sji_map_.wcs for sji_map_ in iris_sji_1400_cube_despike], [dkist_vbi_target_cube_crop_rebin.wcs]*n_iris_sji_1400))

        irissji_1400_pr_dset = np.array([r[0] for r in results_irissji_1400])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/IRISSJI_1400_pr.hdf5","w") as hf:
            hf.create_dataset('irissji_1400_img', data=irissji_1400_pr_dset)

        irissji_1400_nowow_pr_dset = np.array([r[1] for r in results_irissji_1400])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/IRISSJI_1400_nowow_pr.hdf5","w") as hf:
            hf.create_dataset('irissji_1400_nowow_img', data=irissji_1400_nowow_pr_dset)

        irissji_1400_date_obs = [r[2] for r in results_irissji_1400]
        irissji_1400_date_obs_table = Table()
        irissji_1400_date_obs_table["date_obs"] = irissji_1400_date_obs
        irissji_1400_date_obs_table.write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/IRISSJI_1400_date_obs.txt",
                                          format="ascii", overwrite=True)

        del irissji_1400_pr_dset
        del irissji_1400_nowow_pr_dset

        print("IRISSJI_1400 done")

    # AIA 171 plot-ready dataset
    if todo_list["AIA"]:

        aia_171_cube = sunpy.map.Map(sorted(glob("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/AIA/171/*.fits"))[:],sequence=False,
                                    memmap=True)
        
        n_aia_171 = len(aia_171_cube)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_aia_171 = list(executor.map(job_aia_wow_and_reproject, aia_171_cube,
            [aia_map_.wcs for aia_map_ in aia_171_cube], [dkist_vbi_target_cube_crop_rebin.wcs]*n_aia_171))

        aia_171_pr_dset = np.array([r[0] for r in results_aia_171])
        with h5py.File("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/AIA_171_pr.hdf5","w") as hf:
            hf.create_dataset('aia_171_img', data=aia_171_pr_dset)

        aia_171_date_obs = [r[1] for r in results_aia_171]
        aia_171_date_obs_table = Table()
        aia_171_date_obs_table["date_obs"] = aia_171_date_obs
        aia_171_date_obs_table.write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/AIA_171_date_obs.txt",
                                    format="ascii", overwrite=True)

        del aia_171_pr_dset

        print("AIA_171 done")
    
    if todo_list["VBI_DATE_OBS"]:
        Gband_dset_raw.headers[["DATE-AVG",]].write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Gband_AEZDV_date_avg.txt",
                                                    format="ascii", overwrite=True)
        Hbeta_dset_raw.headers[["DATE-AVG",]].write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Hbeta_BJOLO_date_avg.txt",
                                                    format="ascii", overwrite=True)
        CaIIK_dset_raw.headers[["DATE-AVG",]].write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/CaIIK_BZPOW_date_avg.txt",
                                                    format="ascii", overwrite=True)
        TiO_dset_raw.headers[["DATE-AVG",]].write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/TiO_BNRPZ_date_avg.txt",
                                                    format="ascii", overwrite=True)
        Halpha_dset_raw.headers[["DATE-AVG",]].write("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/data/pid_1_123_aux/plot_ready/Halpha_BLZNL_date_avg.txt",
                                                    format="ascii", overwrite=True)

        print("VBI_DATE_OBS done")
