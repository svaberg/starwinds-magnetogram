import numpy as np

import logging
log = logging.getLogger(__name__)
import tempfile

import sys
sys.path.append('/Users/u1092841/Documents/PHD/sw_tools/zdipy/zdipy')

from . import profile_shapes

import lineprofile as zdipy_profile_gaussian
import lineprofileVoigt as zdipy_profile_voigt

log.debug("Successfully imported zdipy line profiles")

lightspeed = 2.99792458e8  # speed of light in m/s


def wavelength_nm_to_velocity_kms(wl, wl0):
    wl = np.asarray(wl)

    velocity_kms = 1e-3 * lightspeed * (wl-wl0) / wl0
    return velocity_kms


def velocity_kms_to_wavelength_nm(vel, wl0):
    velocity_kms = np.asarray(vel)
    wlmwl0dwl0 = velocity_kms * 1e3 / lightspeed
    wl = wlmwl0dwl0 * wl0 + wl0
    return wl


class ZdipyVoigt(profile_shapes.ProfileShape):
    """
    Zdipy Gaussian profile, unknown properties
    It seems that it has strength 1 when sigma = 1 and gamma = 1e-3
    """
    def __init__(self, sigma, gamma):
        super().__init__()
        _fake_center_wl = 250.0
        _fake_strength = 1.0
        self.profile = lineData(_fake_center_wl, _fake_strength, sigma, gamma, 0, 0)

    def __call__(self, v_kms):
        wl_nm = velocity_kms_to_wavelength_nm(v_kms, self.profile.wl0)
        if wl_nm.shape != ():
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(wl_nm), wl_nm)
            retval = 1 - temp.Iunscaled
        elif wl_nm.shape == ():
            fake_len = 99
            fake_center = 49
            fake_wl = np.linspace(0.5, 1.5, fake_len) * wl_nm
            assert(fake_wl[fake_center] == wl_nm)
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(fake_wl), fake_wl)
            retval = 1 - temp.Iunscaled[fake_center]

        return retval

    def __str__(self):
        return "Zdipy Voigt"


class ZdipyOneParam(profile_shapes.ProfileShape):
    """
    Zdipy profile, unknown properties
    """
    def __init__(self, gamma_sigma):
        super().__init__()
        _fake_center_wl = 250.0
        _fake_strength = 1.0
        self.profile = lineData(_fake_center_wl, _fake_strength, 1, gamma_sigma, 0, 0)

    def __call__(self, v_kms):
        wl_nm = velocity_kms_to_wavelength_nm(v_kms, self.profile.wl0)
        if wl_nm.shape != ():
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(wl_nm), wl_nm)
            retval = 1 - temp.Iunscaled
        elif wl_nm.shape == ():
            fake_len = 99
            fake_center = 49
            fake_wl = np.linspace(0.5, 1.5, fake_len) * wl_nm
            assert(fake_wl[fake_center] == wl_nm)
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(fake_wl), fake_wl)
            retval = 1 - temp.Iunscaled[fake_center]

        return retval

    def __str__(self):
        return __name__


class lineData:
    def __init__(self, wl0, str, widthGauss, widthLorentz, g, limbDark):
        # Read in model line profile data from inFileName, and store it as part of the lineData object
        # lines beginning with a # are ignored (i.e. treated as comments)
        # Warning: currently the local line profile only uses the first (non-comment) line of line data
        # that should be pretty easy to expand later for multi-line profiles/spectra
        self.wl0 = np.array([])
        self.str = np.array([])
        self.widthGauss = np.array([])
        self.widthLorentz = np.array([])
        self.g = np.array([])
        self.limbDark = np.array([])
        self.numLines = 1#len(self.wl0)

        self.wl0 = np.append(self.wl0, wl0)
        self.str = np.append(self.str, str)
        self.widthGauss = np.append(self.widthGauss, widthGauss)
        self.widthLorentz = np.append(self.widthLorentz, widthLorentz)
        self.g = np.append(self.g, g)
        self.limbDark = np.append(self.limbDark, limbDark)
