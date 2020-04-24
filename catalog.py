from astropy.coordinates import SkyCoord
from astropy import units as u, coordinates as coord
from astroquery.vizier import Vizier
import lib_fits as lib_fits
import numpy as np
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, imshow_norm
from astropy.visualization import SqrtStretch
import matplotlib.pyplot as plt

#header, image = lib_fits.read_image('/home/paduverne/Documents/ATLAS18qqn-S001-R001-C001-SDSS_g_ref_g.fits')
#header, image = lib_fits.read_image('/home/paduverne/Documents/Stage M2/IRIS/ATLAS18qqn-S001-R001-C001-SDSS_g.fits')
#header, image = lib_fits.read_image('/home/paduverne/Documents/data/20180621/ATLAS18qqn-S001-R001-C001-SDSS_z.fits')
#header, image = lib_fits.read_image('/home/paduverne/Documents/Entrainement_calibration/14bv_t60_les_makes/20190814/gw190814f1.fit')
#c = coord.SkyCoord(RA, dec, unit=(u.hourangle, u.deg),frame='icrs')
#ra = header['RA']
#dec = header['DEC']
#rad_deg = 2. * u.deg
#field = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
#gaia = Vizier.query_region(field, width=rad_deg, catalog="I/345/gaia2")[0]
#usno = Vizier.query_region(field, width=rad_deg, catalog="I/284/out")[0]
#sdds = Vizier.query_region(field, width=rad_deg, catalog="V/147/sdss12")[0]
#ps = Vizier.query_region(field, width=rad_deg, catalog="II/349/ps1")[0]



def catalog_choice(header):
    """
    Process a choice for the catalog to do photometric calibration
    
    parameters: 
        header: sheader of an image to get necessary informations

    returns: two string, name of catalog to use and name of the filter 
             given in header
    """
    try:
        ra = header['RA']
    except Exception:
        print('No Ra found in header, proceed to astrometric calibration')
    try:
        dec = header['DEC']
    except Exception:
        print('No Dec found in header, proceed to astrometric calibration')
    try:
        filter = header['FILTER']
    except Exception:
        print('No filter found in header')
    rad_deg = 1. * u.deg
    field = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    DEC = header['CRVAL2']
    z_band = ['z', "z'", 'sdss z', 'SDSS z']
    if float(DEC) > -30.:
        catalog = 'II/349/ps1' #Use of Pan-STARRS DR 1 as much as possible
        print(catalog)
    elif Vizier.query_region(field, width=rad_deg, height=rad_deg,
                             catalog="V/147/sdss12")[0]:
        catalog = "V/147/sdss12" #Try SDSS if pan-STARRS is not available
        print(catalog)
    elif filter not in z_band:
        catalog = "I/345/gaia2"#It is impossible to convert GAIA's bands into
                               #SDSS z-band. Otherwise, as GAIA is an all sky
                               #catalog, it is a good option to calibrate.
        print(catalog)
    else:
        catalog = "I/284/out" #Last resort is USNO B1
        print(catalog)
    return catalog, filter

def poly(x, coefficients):
     """
     Compute a polynome, useful for transformation laws
     
     parameters: 
         x: Variable for the polynome
         coefficients: list of coefficients
         
     returns: float result of the polynome
     """
     poly = 0
     for i, coef in enumerate(coefficients):
         poly += coef * x**i

     return poly
 
def from_gaia2Johnson(filter, Gaia_Table):
    """
    Give the transformation laws to go from gaia photometric system 
    to Johnson-Cousins photometric system
    Transformation given by https://gea.esac.esa.int/archive/documentation/
    GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/
    ssec_cu5pho_PhotTransf.html
    Table 5.8
    WARNING : GAIS's Magnitudes are in the Vega system
    
    parameters: filter: filter used to get the image
                Gaia_Table: astropy.table with information from Gaia for 
                the sources in the image
                
    returns: astropy.table with informations from Gaia and the computation 
             for the filter of the image
    """
    V_band = ['V'] #Differen15.690t names that can be 
    R_band = ['R'] #found in header
    I_band = ['I']
    B_band = ['B']
    
    if filter in V_band:
        mask = (-0.5 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 2.75)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [-0.01760, -0.006860, -0.1732]
        Result["VMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)
        Result["error_from_transformation"] = 0.045858

    if filter in R_band:
        mask = (-0.5 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 2.75)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [-0.003226, 0.3833, -0.1345]
        Result["RMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)
        Result["error_from_transformation"] = 0.04840

    if filter in I_band:
        mask = (-0.5 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 2.75)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [0.02085, 0.7419, -0.09631]
        Result["IMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)  ##########Revoir cette trabsformation, pt etre mal  implementee
        Result["error_from_transformation"] = 0.04956

    if filter in B_band:
        raise ValueError('No transformation law for Gaia->B_Jonhson')
        
    return Result
    
def from_gaia2SDSS(filter, Gaia_Table):
    """
    Give the transformation laws to go from gaia photometric system 
    to SDSS photometric system
    Transformation given by https://gea.esac.esa.int/archive/documentation/
    GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/
    ssec_cu5pho_PhotTransf.html
    Table 5.7
    WARNING : GAIS's Magnitudes are in the Vega system
    
    parameters: filter: filter used to get the image
                Gaia_Table: astropy.table with information from Gaia for 
                the sources in the image
                
    returns: astropy.table with informations from Gaia and the computation 
             for the filter of the image
    """
    g_band = ['g', "g'", 'sdss g', 'SDSS g'] #Different names that can be 
    r_band = ['r', "r'", 'sdss r', 'SDSS r'] #found in header
    i_band = ['i', "i'", 'sdss i', 'SDSS i']
    z_band = ['z', "z'", 'sdss z', 'SDSS z']
    
    if filter in r_band:
        mask = (0.2 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 2.7)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [-0.12879, 0.24662, -0.027464, -0.049465]
        Result["r_SDSSMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)
        Result["error_from_transformation"] = 0.066739

    if filter in i_band:
        mask = (0.0 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 4.5)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [-0.29676, 0.64728, -0.10141, 0.]
        Result["i_SDSSMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)
        Result["error_from_transformation"] = 0.098957

    if filter in g_band:
        mask = (-0.5 < Gaia_Table['bp_rp'])
        Result = Gaia_Table[mask]
        mask = (Result['bp_rp'] < 2.0)
        Result = Result[mask] #Validity of the transformation law
        coefficients = [0.13518, -0.46245, -0.25171, 0.021349]
        Result["g_SDSSMag"] = Result["phot_g_mean_mag"] - poly(Result['bp_rp'],
                                                            coefficients)
        Result["error_from_transformation"] = 0.16497

    if filter in z_band:
        raise ValueError('No transformation law for Gaia->z_sdss')
        
    return Result

def from_usno2Johnson(filter, USNO_Table):
    """
    Give the transformation laws to go from USNO photometric system 
    to Johnson photometric system
    Transformation given by http://www.mpe.mpg.de/~jcg/GROND/calibration.html
    
    parameters: filter: filter used to get the image
                USNO_Table: astropy.table with information from USNO for 
                the sources in the image
                
    returns: astropy.table with informations from USNO and the computation 
             for the filter of the image
    """
    V_band = ['V'] #Different names that can be 
    R_band = ['R'] #found in header
    I_band = ['I']
    B_band = ['B']
    
    if filter in V_band:
        USNO_Table['V_JohnsonMag'] = 0.444 * USNO_Table['B1mag'] + 0.556 * USNO_Table['R1mag']
        USNO_Table["error_from_transformation"] = 0.5
    if filter in B_band:
        USNO_Table['B_JohnsonMag'] = USNO_Table['B1mag']
        USNO_Table["error_from_transformation"] = 0.5
    if filter in R_band:
        USNO_Table['R_JohnsonMag'] = USNO_Table['R1mag']
        USNO_Table["error_from_transformation"] = 0.5
    if filter in I_band:
        USNO_Table['I_JohnsonMag'] = USNO_Table['Imag']
        USNO_Table["error_from_transformation"] = 0.5
#        raise ValueError('No transformation law known for USNO->I_Johnson')

    return USNO_Table

def from_SDSS2Johnson(filter, SDSS_Table):
    """
    Give the transformation laws to go from SDSS photometric system 
    to Johnson photometric system
    Transformation given by http://www.sdss3.org/dr8/algorithms/
    sdssUBVRITransform.php
    
    parameters: filter: filter used to get the image
                SDSS_Table: astropy.table with information from SDSS for 
                the sources in the image
                
    returns: astropy.table with informations from SDSS and the computation 
             for the filter of the image
    """
    V_band = ['V'] #Different names that can be 
    R_band = ['R'] #found in header
    I_band = ['I']
    B_band = ['B']
    
    if filter in V_band:
        coefficients = [- 0.016, -0.573]
        SDSS_Table['g-r'] = SDSS_Table['gmag'] - SDSS_Table['rmag']
        SDSS_Table["VMag"] = SDSS_Table["gmag"] + poly(SDSS_Table['g-r'],
                                                            coefficients)
        SDSS_Table["error_from_transformation"] = np.sqrt((0.002 * SDSS_Table['g-r'])**2 + 0.573**2 * SDSS_Table['e_rmag']**2 + 0.427**2 * SDSS_Table['e_gmag']**2 + 0.002**2)
        
    if filter in R_band:
        coefficients = [0.152, -0.257]
        SDSS_Table['r-i'] = SDSS_Table['rmag'] - SDSS_Table['imag']
        SDSS_Table["RMag"] = SDSS_Table["rmag"] + poly(SDSS_Table['r-i'],
                                                            coefficients)
        SDSS_Table["error_from_transformation"] = np.sqrt(0.743**2 * SDSS_Table['e_rmag']**2 + 0.257**2 * SDSS_Table['e_imag']**2 + 0.002**2 + SDSS_Table['r-i']**2 * 0.004**2)
        
    
    if filter in I_band:
        coefficients = [-0.394, -0.409]
        SDSS_Table['i-z'] = SDSS_Table['imag'] - SDSS_Table['zmag']
        SDSS_Table["IMag"] = SDSS_Table["imag"] + poly(SDSS_Table['i-z'],
                                                            coefficients)
        SDSS_Table["error_from_transformation"] = np.sqrt(0.591**2 * SDSS_Table['e_imag']**2 + 0.409**2 * SDSS_Table['e_zmag']**2 + 0.002**2 + SDSS_Table['i-z']**2 * 0.006**2)
    
    if filter in B_band:
        coefficients = [0.219, 0.312]
        SDSS_Table['g-r'] = SDSS_Table['gmag'] - SDSS_Table['rmag']
        SDSS_Table["BMag"] = SDSS_Table["gmag"] + poly(SDSS_Table['g-r'],
                                                            coefficients)
        SDSS_Table["error_from_transformation"] = np.sqrt(1.312**2 * SDSS_Table['e_gmag']**2 + 0.312**2 * SDSS_Table['e_rmag']**2 + 0.002**2 + SDSS_Table['g-r']**2 * 0.003**2)
        
    return SDSS_Table
    

def from_SDSSdr82Johnson(filter, SDSS_Table):
    """
    Give the transformation laws to go from SDSS photometric system 
    to Johnson photometric system
    Transformation given by http://www.sdss3.org/dr8/algorithms/
    sdssUBVRITransform.php
    
    parameters: filter: filter used to get the image
                SDSS_Table: astropy.table with information from SDSS for 
                the sources in the image
                
    returns: astropy.table with informations from SDSS and the computation 
             for the filter of the image
    """
    V_band = ['V'] #Different names that can be 
    R_band = ['R'] #found in header
    I_band = ['I']
    B_band = ['B']
    
    # if filter in V_band:
    #     coefficients = [- 0.016, -0.573]
    #     SDSS_Table['g-r'] = SDSS_Table['gmag'] - SDSS_Table['rmag']
    #     SDSS_Table["VMag"] = SDSS_Table["gmag"] + poly(SDSS_Table['g-r'],
    #                                                         coefficients)
    #     SDSS_Table["error_from_transformation"] = np.sqrt((0.002 * SDSS_Table['g-r'])**2 + 0.573**2 * SDSS_Table['e_rmag']**2 + 0.427**2 * SDSS_Table['e_gmag']**2 + 0.002**2)
        
    if filter in R_band:
        coefficients = [0.152, -0.257]
        SDSS_Table['r-i'] = SDSS_Table['r'] - SDSS_Table['i']
        SDSS_Table["RMag"] = SDSS_Table["r"] + poly(SDSS_Table['r-i'],
                                                            coefficients)
        SDSS_Table["error_from_transformation"] = np.sqrt(0.743**2 * SDSS_Table['r_err']**2 + 0.257**2 * SDSS_Table['i_err']**2 + 0.002**2 + SDSS_Table['r-i']**2 * 0.004**2)
        
    
    # if filter in I_band:
    #     coefficients = [-0.394, -0.409]
    #     SDSS_Table['i-z'] = SDSS_Table['imag'] - SDSS_Table['zmag']
    #     SDSS_Table["IMag"] = SDSS_Table["imag"] + poly(SDSS_Table['i-z'],
    #                                                         coefficients)
    #     SDSS_Table["error_from_transformation"] = np.sqrt(0.591**2 * SDSS_Table['e_imag']**2 + 0.409**2 * SDSS_Table['e_zmag']**2 + 0.002**2 + SDSS_Table['i-z']**2 * 0.006**2)
    
    # if filter in B_band:
    #     coefficients = [0.219, 0.312]
    #     SDSS_Table['g-r'] = SDSS_Table['gmag'] - SDSS_Table['rmag']
    #     SDSS_Table["BMag"] = SDSS_Table["gmag"] + poly(SDSS_Table['g-r'],
    #                                                         coefficients)
    #     SDSS_Table["error_from_transformation"] = np.sqrt(1.312**2 * SDSS_Table['e_gmag']**2 + 0.312**2 * SDSS_Table['e_rmag']**2 + 0.002**2 + SDSS_Table['g-r']**2 * 0.003**2)
        
    return SDSS_Table 
        
def from_PS2Johnson(filter, PS_Table):
        """
        Give the transformation laws to go from Pan-STARRS photometric system 
        to Johnson photometric system
        Transformation given by http://www.sdss3.org/dr8/algorithms/
        sdssUBVRITransform.php
        
        parameters: filter: filter used to get the image
                    PS_Table: astropy.table with information from PS for 
                    the sources in the image
                    
        returns: astropy.table with informations from PS and the computation 
                 for the filter of the image
        """
        V_band = ['V'] #Different names that can be 
        R_band = ['R'] #found in header
        I_band = ['I']
        B_band = ['B']
        
        if filter in V_band:
            coefficients = [- 0.016, -0.573]
            PS_Table['g-r'] = PS_Table['gmag'] - PS_Table['rmag']
            PS_Table["VMag"] = PS_Table["gmag"] + poly(PS_Table['g-r'],
                                                                coefficients)
            PS_Table["error_from_transformation"] = np.sqrt((0.002 * PS_Table['g-r'])**2 + 0.573**2 * PS_Table['e_rmag']**2 + 0.427**2 * PS_Table['e_gmag']**2 + 0.002**2)
            
        if filter in R_band:
            coefficients = [0.152, -0.257]
            PS_Table['r-i'] = PS_Table['rmag'] - PS_Table['imag']
            PS_Table["RMag"] = PS_Table["rmag"] + poly(PS_Table['r-i'],
                                                                coefficients)
            PS_Table["error_from_transformation"] = np.sqrt(0.743**2 * PS_Table['e_rmag']**2 + 0.257**2 * PS_Table['e_imag']**2 + 0.002**2 + PS_Table['r-i']**2 * 0.004**2)
            
        
        if filter in I_band:
            coefficients = [-0.394, -0.409]
            PS_Table['i-z'] = PS_Table['imag'] - PS_Table['zmag']
            PS_Table["IMag"] = PS_Table["imag"] + poly(PS_Table['i-z'],
                                                                coefficients)
            PS_Table["error_from_transformation"] = np.sqrt(0.591**2 * PS_Table['e_imag']**2 + 0.409**2 * PS_Table['e_zmag']**2 + 0.002**2 + PS_Table['i-z']**2 * 0.006**2)
        
        if filter in B_band:
            coefficients = [0.219, 0.312]
            PS_Table['g-r'] = PS_Table['gmag'] - PS_Table['rmag']
            PS_Table["BMag"] = PS_Table["gmag"] + poly(PS_Table['g-r'],
                                                                coefficients)
            PS_Table["error_from_transformation"] = np.sqrt(1.312**2 * PS_Table['e_gmag']**2 + 0.312**2 * PS_Table['e_rmag']**2 + 0.002**2 + PS_Table['g-r']**2 * 0.003**2)
            
        return PS_Table

def from_luminance2PS(filter, PS_Table):
    
    L_band = ['L', 'l', 'luminance']
    coefficients = [0.297]
    # 0.08<(r−i)<0.5 and 0.2<(g−r)<1.4 
    # mask = (0.0 < Gaia_Table['bp_rp'])
    #     Result = Gaia_Table[mask]
    #     mask = (Result['bp_rp'] < 4.5)
    PS_Table['r-i'] = PS_Table['rmag'] - PS_Table['imag']
    PS_Table['g-r'] = PS_Table['gmag'] - PS_Table['rmag']
    mask = (0.08 < PS_Table['r-i'])
    PS_Table = PS_Table[mask]
    mask = (PS_Table['r-i'] < 0.5)
    PS_Table = PS_Table[mask]
    mask = (0.2 < PS_Table['g-r'])
    PS_Table = PS_Table[mask]
    mask = (PS_Table['g-r'] < 1.4)
    PS_Table = PS_Table[mask]
    if filter in L_band:
        PS_Table['LMag'] = coefficients[0]* PS_Table["gmag"] + (1-coefficients[0]) * PS_Table['rmag']
        
    return PS_Table

def from_PS2lumi_coarse(filter, PS_Table):
    
    L_band = ['L', 'l', 'luminance']
    if filter in L_band:
        PS_Table['flux_r'] = 3631 * 10**(-0.4 * PS_Table["rmag"])
        PS_Table['flux_g'] = 3631 * 10**(-0.4 * PS_Table["gmag"])
        PS_Table['lmag'] = -2.5 * np.log10((PS_Table['flux_r'] + PS_Table['flux_g']) / 3631)
        
    return PS_Table

def from_truePS2Johnson(filter, PS_Table):
    
    R_band = ['R']
    if filter in R_band:
        coefficients = [-0.163, -0.086, -0.061]
        PS_Table['g-r'] = PS_Table['gmag'] - PS_Table['rmag']
        PS_Table["RMag"] = PS_Table["rmag"] + poly(PS_Table['g-r'], coefficients)
        PS_Table["error_from_transformation"] = 0.1
        
    return PS_Table
    
    
def from_Vega2AB(filter, mag_table):
    """
    Give the transformation laws to go from Vega system to AB system
    Transformation given by http://www.astronomy.ohio-state.edu/
    ~martini/usefuldata.html
    Based on Blanton et al. (2007)
    
    parameters: filter: filter used to get the image
                mag_Table: astropy.table with magnitued in Vega system
                
    returns: astropy.table with informations from PS and the computation 
             for the filter of the image
    """
    V_band = ['V'] #Different names that can be 
    R_band = ['R'] #found in header
    I_band = ['I']
    B_band = ['B']
    g_band = ['g', "g'", 'sdss g', 'SDSS g'] #Different names that can be 
    r_band = ['r', "r'", 'sdss r', 'SDSS r'] #found in header
    i_band = ['i', "i'", 'sdss i', 'SDSS i']
    z_band = ['z', "z'", 'sdss z', 'SDSS z']
    
    #Conversion for Johnson-Cousins photometric System
    if filter in V_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.02
    if filter in B_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] - 0.09
    if filter in R_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.21
    if filter in I_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.45

    #Conversion for sdss/Pan-STARRS photometric System
    if filter in g_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] - 0.08
    if filter in r_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.16
    if filter in i_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.37
    if filter in z_band:
        mag_table['magnitude_AB'] = mag_table['magnitude'] + 0.54
    
    return mag_table



#def from_usno2SDSS(filter, USNO_Table):
#    
#    g_band = ['g', "g'", 'sdss g', 'SDSS g'] #Different names that can be 
#    r_band = ['r', "r'", 'sdss r', 'SDSS r'] #found in header
#    i_band = ['i', "i'", 'sdss i', 'SDSS i']
#    z_band = ['z', "z'", 'sdss z', 'SDSS z']
#    
#    if filter in g_band:
#        USNO_Table = from_usno2Johnson('B', USNO_Table)
#        USNO_Table["error_1"] = USNO_Table["error_from_transformation"]
#        USNO_Table = from_usno2Johnson('V', USNO_Table)
#        USNO_Table["error_2"] = USNO_Table["error_from_transformation"]
#        coefficients = [-0.127, 0.634]
#        USNO_Table["B-V"] = USNO_Table["BMag"] - USNO_Table["VMag"]
#        USNO_Table["gMag"] = USNO_Table["VMag"] + poly(USNO_Table["B-V"], coefficients)
#        USNO_Table["error_from_transformation"] = np.sqrt(USNO_Table["error_1"]**2 + USNO_Table["error_2"]**2 + ) -> a discuter aussi
#        g-V   =     (0.634 ± 0.002)*(B-V)  - (0.127 ± 0.002)
        

# A voir pour la suite de cette fonction, on ne peut pas avoir i et z avec USNO!     
        
#        r-R   =     (0.275 ± 0.006)*(V-R)  + (0.086 ± 0.004) if V-R <= 0.93
#        r-R   =     (0.71 ± 0.05)*(V-R)    - (0.31 ± 0.05)   if V-R >  0.93
    
    
    
    
    
    
    
    
#####
#Etoiles de verification

#coordinate = [243.9750953, 22.2772574] #etoile 1 -> a cote de atlas cow
#coordinate = [244.1670864, 22.3581269] #etoile 2
#
#w = WCS(header)
#coordinates = coord.SkyCoord(coordinate[0], coordinate[1],
#                                 unit=(u.deg, u.deg), frame='icrs')
#coordinates = np.array([[coordinates.ra.deg, coordinates.dec.deg]])
#coord_image = w.all_world2pix(coordinates, 1)
#xs=coord_image[0][0]
#ys=coord_image[0][1]
#rad_deg = 7. * u.arcsec
#field1 = SkyCoord(coordinate[0], coordinate[1], unit=(u.deg, u.deg), frame='icrs')
#gaia1 = Vizier.query_region(field1, width=rad_deg, catalog="I/345/gaia2")[0]
#usno1 = Vizier.query_region(field1, width=rad_deg, catalog="I/284/out")[0]
#sdds1 = Vizier.query_region(field1, width=rad_deg, catalog="V/147/sdss12")[0]
#ps1 = Vizier.query_region(field1, width=rad_deg, catalog="II/349/ps1")[0]
#
#sdss2 = from_gaia2SDSS('g', gaia1)
#im, norm = imshow_norm(image,  origin='lower', cmap = 'Greys',
#                       interval=ZScaleInterval(), stretch=SqrtStretch())
#sdss3 = from_gaia2SDSS('r', gaia1)
#sdss4 = from_gaia2SDSS('i', gaia1)
#
#us1 = from_gaia2Johnson('V', gaia1)
##us2 = from_gaia2Johnson('B', gaia1)
#
#ps2 = from_PS2Johnson('V', ps1)
#ps3 = from_PS2Johnson('R', ps1)
#ps4 = from_PS2Johnson('B', ps1)
#plt.scatter(xs, ys)
#####


    
    
    
    
    
    
    
    
    
    