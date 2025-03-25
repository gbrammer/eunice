import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import astropy.coordinates
import astropy.units as u
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from grizli.aws import db
from grizli import utils

def mode_statistic(data):
    return 2.5*np.median(data) - 1.5*np.mean(data)

def edge_dq():
    dq = np.zeros((2048, 2048))
    dq[:4,:] |= 1
    dq[-4:,:] |= 1
    dq[:,:4] |= 1
    dq[:, -4:] |= 1
    return dq
    

def make_output_filename(file):
    out_file = os.path.join(
        os.path.dirname(file),
        "flat_" + os.path.basename(file).lower(),
    )
    
    return out_file
    
def nisp_raw_pipeline(raw_file, force=False, verbose=True):
    """
    """
    
    out_file = make_output_filename(raw_file)
    utils.LOGFILE = out_file.replace(".fits", ".log.txt")
    
    if os.path.exists(out_file) & (not force):
        print("found " + out_file)
        return out_file
        
    msg = f"nisp_raw_pipeline {raw_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    out_file, fcorr = process_raw_file(raw_file, verbose=verbose)
    fcorr.writeto(out_file, overwrite=True)
    
    gaia_catalog = nisp_gaia_catalog(raw_file)
    gaia = utils.read_catalog(gaia_catalog)
    
    for i in range(4):
        for j in range(4):
            plt.close('all')
        
            det = f'{i+1}{j+1}'
            ext = f'DET{det}.SCI'
            utils.log_comment(utils.LOGFILE, f"Align detector {ext}", verbose=True)
        
            _ = align_nisp_detector(
                fcorr[ext].data,
                fcorr[ext].header,
                gaia,
                threshold=10,
                verbose=verbose
            )
    
    fcorr.writeto(out_file, overwrite=True)
    
    return out_file
    
def process_raw_file(file="EUC_LE1_NISP-03015-1-D_20240805T205608.000000Z_01_03_01.00.fits", flat_limits=[0.9, 1.05], verbose=True):
    """
    """
    out_file = make_output_filename(file)
    
    reg_file = out_file.replace(".fits", ".reg")
    with open(reg_file, "w") as fp:
        fp.write("fk5\n")
        
    with pyfits.open(file) as img:
        hdul = pyfits.HDUList([
            pyfits.PrimaryHDU(header=img[0].header)
        ])
        
        with open(reg_file, "a") as fp:
            fp.write(f"point({img[0].header['RA']},{img[0].header['DEC']}) # point=x\n")

        for i in range(4):
            for j in range(4):
                flat_file= f"euc_le1_nisp_flat_h_det{i+1}{j+1}.fits".lower()

                flat_path = os.path.join(
                    "/Users/gbrammer/Research/JWST/Projects/Euclid",
                    flat_file
                )
                
                flat_img = pyfits.open(flat_path)
                iflat = 1./flat_img[0].data
                bad = (flat_img[0].data < flat_limits[0])
                bad |= (flat_img[0].data > flat_limits[1])
                
                iflat[flat_img[0].data <= 0] = 0
                iflat[bad] = 0
                
                ext = f'DET{i+1}{j+1}.SCI'
                
                msg = f"extension {ext}  {flat_file}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                
                fcorr = img[ext].data * iflat

                # mask = (err.astype(float) <= 0)

                rms = utils.nmad(fcorr[iflat > 0])
                
                hext = img[ext].header
                hext["FLATFILE"] = flat_file
                hext["FLATLOW"] = flat_limits[0]
                hext["FLATHIGH"] = flat_limits[1]
                hext["NMAD_ERR"] = rms
                hext["MDRIZSKY"] = mode_statistic(fcorr[iflat > 0])
                
                wcs_header = initialize_detector_wcs_header(
                    img[0].header, detector=f'{i+1}{j+1}'
                )
                for k in wcs_header:
                    hext[k] = wcs_header[k]

                hdul.append(pyfits.ImageHDU(header=hext, data=fcorr.astype(np.float32)))
                
                wcs_i = pywcs.WCS(hext, relax=True)
                sr = utils.SRegion(wcs_i)
                with open(reg_file, "a") as fp:
                    sr.label = ext
                    fp.write(sr.region[0] + "\n")
                
                flat_img.close()
    
    return out_file, hdul


def initialize_detector_wcs_header(header, detector='11'):
    """
    """
    wtab = utils.read_catalog(
        os.path.join(os.path.dirname(__file__), 'data/euclid_nisp_detector_offsets.csv')
    )
    
    # OSS reference coordinate
    oss_ref = astropy.coordinates.SkyCoord(header['RA'], header['DEC'], unit='deg')

    pa_rad = header['PA']/180*np.pi

    crval = oss_ref.directional_offset_by(
        (wtab['offset_pa'] + pa_rad)*u.radian,
        wtab['offset_sep']*u.deg
    )
    
    ix = np.where(wtab['ext'] == f'DET{detector}.SCI')[0][0]

    wtab['crval1'] = crval.ra.deg
    wtab['crval2'] = crval.dec.deg

    head, xw = utils.make_wcsheader(
        ra=wtab['crval1'][ix],
        dec=wtab['crval2'][ix],
        size=2048*wtab['pscale'][ix],
        pixscale=wtab['pscale'][ix],
        theta=(header['PA'] + wtab['pa'][ix] - 90),
        get_hdu=False
    )

    # CD matrix and SIP coefficients
    for k in wtab.colnames:
        if k.startswith('cd'):
            head[k.upper()] = wtab[k][ix]

    # Rotate by OSS PA
    hwcs = utils.transform_wcs(
        pywcs.WCS(head),
        rotation=header['PA']/180*np.pi
    )
    head = utils.to_header(hwcs)

    for k in wtab.colnames:
        if k.startswith('cd') | k.startswith('crv') | k.startswith('offset_'):
            continue
        
        head[k.upper()] = wtab[k][ix]
    
    return head
    
def nisp_gaia_catalog(file="EUC_LE1_NISP-03015-1-D_20240805T205608.000000Z_01_03_01.00.fits"):
    """
    """
    import grizli.catalog

    gaia_file = os.path.join(
        os.path.dirname(file),
        os.path.basename(file).lower().replace('.fits', '.gaia.fits'),
    )
    
    if os.path.exists(gaia_file):
        return gaia_file

    with pyfits.open(file) as img:
        header = img[0].header
    
    gaia = grizli.catalog.get_gaia_vizier(
        ra=header['RA'],
        dec=header['DEC'],
        radius=32,
        mjd=header['MJD-OBS'],
        clean_mjd=True,
        verbose=False,
    )
    
    gaia.write(gaia_file, overwrite=True)
    
    print('GAIA: ', len(gaia))
    
    return gaia_file
    
def align_nisp_detector(data, header, gaia, threshold=3.0, n_iter=3, align_threshold=5, verbose=True, update=True):
    """
    """
    from skimage.transform import SimilarityTransform
    from tristars import match
    
    import sep
    from grizli import prep
    
    sep.set_extract_pixstack(3e6)
    
    mask = (data <= 0.0) | (~np.isfinite(data))
    bkg = np.nanmedian(data[~mask])
    
    nmad = utils.nmad(data[~mask])
    
    err = np.ones(data.shape, dtype=np.float32)

    c, cseg = prep.make_SEP_catalog_from_arrays(
        data - bkg, # .byteswap().newbyteorder(),
        err,
        mask,
        threshold=threshold,
        get_background=False,
        segmentation_map=False,
    )
    
    print(f'bkg: {bkg:.2f}  N={len(c)}')
    
    translation = np.zeros(2)
    rotation = 0.
    scale = 1.0
    
    for _iter in range(n_iter):
        wcs = pywcs.WCS(header, relax=True)
        wcs = utils.transform_wcs(
            wcs,
            translation=-translation, rotation=-rotation, scale=1./scale
        )
        
        c['ra'], c['dec'] = wcs.all_pix2world(c['x'], c['y'], 0)
        
        idx, dr, dx, dy = gaia.match_to_catalog_sky(
            c, self_radec=('ra_time','dec_time'),
            get_2d_offset=True
        )
        
        hasm = dr.value < align_threshold

        thresh = np.maximum(align_threshold / 2, 1)

        V1 = np.array([c['x'][hasm], c['y'][hasm]]).T - 1023.5

        gxy = wcs.all_world2pix(gaia['ra_time'], gaia['dec_time'], 0)

        ii = np.unique(idx[hasm])
        V2 = np.array([gxy[0][ii], gxy[1][ii]]).T - 1023.5

        ba_max = 0.9
        size_limit = [5,1800]

        pair_ix = match.match_catalog_tri(
            V1, V2,
            maxKeep=4, auto_keep=3, ignore_rot=False,
            ba_max=ba_max, size_limit=size_limit
        )

        tfo, dx, rms = match.get_transform(
            V1, V2, pair_ix, transform=SimilarityTransform, use_ransac=True
        )
        fig = match.match_diagnostic_plot(V1, V2, pair_ix, tf=tfo,
                                          new_figure=True)
        
        translation += tfo.translation
        rotation += tfo.rotation
        scale *= tfo.scale
        
        msg = f"Iter {_iter}: ({tfo.translation[0]:10.4f}, {tfo.translation[1]:10.4f}) "
        msg += f" {tfo.rotation:7.4f}  {tfo.scale:7.4f}"
        msg += f"  rms = {rms[0]:.2f}, {rms[1]:.2f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    wcs = pywcs.WCS(header, relax=True)
    wcs = utils.transform_wcs(
        wcs,
        translation=-translation, rotation=-rotation, scale=1./scale
    )
    
    c['ra'], c['dec'] = wcs.all_pix2world(c['x'], c['y'], 0)
    if update:
        header['ALIGNDX'] = translation[0]
        header['ALIGNDY'] = translation[1]
        header['ALIGNROT'] = rotation
        header['ALIGNSCL'] = scale
        corr_header = utils.to_header(wcs)
        for k in corr_header:
            header[k] = corr_header[k]
            
        
        
    return translation, rotation, scale, c, wcs
    
    