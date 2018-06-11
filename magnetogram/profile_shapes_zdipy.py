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
        self.sigma = sigma
        self.gamma = gamma
        self._fake_center_wl = 250
        self._fake_strength = 1
        with tempfile.NamedTemporaryFile(mode='w') as temp:
            print(temp.name)

            temp.write("# wl0    str     GaussWidth(km/s)  LorantzWidth(1/gausswidth)  lande_g  limb_darkening\n")
            print(sigma, gamma)
            temp.write("%f  %f  %f  %f  %f  %f" % (self._fake_center_wl, self._fake_strength, sigma, gamma, 0, 0))

            temp.seek(0)  # 'rewind' the file?

            self.profile = zdipy_profile_voigt.lineData(temp.name)
            print(self.profile.__dict__)

    def __call__(self, v_kms):
        wl_nm = velocity_kms_to_wavelength_nm(v_kms, self._fake_center_wl)
        wl_nm = np.asarray(wl_nm)
        if wl_nm.shape != ():
            # print(min(wl_nm), max(wl_nm))
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(wl_nm), wl_nm)
            retval = 1 - temp.Iunscaled
        elif wl_nm.shape == ():
            # raise NotImplementedError("oops")
            fake_len = 99
            fake_center = 49
            fake_wl = np.linspace(0.5, 1.5, fake_len) * wl_nm
            assert(fake_wl[fake_center] == wl_nm)
            temp = zdipy_profile_voigt.localProfileAndDeriv(self.profile, len(fake_wl), fake_wl)
            retval = 1 - temp.Iunscaled[fake_center]

        return retval


    def __str__(self):
        return "Zdipy Gaussian"


