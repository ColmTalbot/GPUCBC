import numpy as np

try:
    import cupy as xp
except ImportError:
    xp  = np

import lal
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.core.utils import create_frequency_series

GAMMA_E = lal.GAMMA

class TF2(object):
    """
    A toy copy of the TaylorF2 waveform.

    This has not been tested. Beware.
    """
    
    def __init__(self, m1, m2, chi1, chi2):
        m1 *= lal.MSUN_SI * lal.G_SI / lal.C_SI**3
        m2 *= lal.MSUN_SI * lal.G_SI / lal.C_SI**3
        self.m1 = m1
        self.m2 = m2
        self.chi1 = chi1
        self.chi2 = chi2
        self.mtot = m1 + m2
        self.eta = m1 * m2 / self.mtot**2
        self.delta = (self.m1 - self.m2) / self.mtot
        self.chis = (self.chi1 + self.chi2) / 2
        self.chia = (self.chi1 - self.chi2) / 2
        self.chi_pn = (
            (self.m1 * self.chi1 + self.m2 * self.chi2) / self.mtot -
            38 * self.eta / 113 * (self.chi1 + self.chi2))
        seta = np.sqrt(1.0 - 4.0 * self.eta)
        m1 = 0.5 * (1.0 + seta)
        m2 = 0.5 * (1.0 - seta)
        self.chi = (m1**2 * self.chi1 + m2**2 * self.chi2)
        self.fisco = 6**(-1.5) / np.pi / self.mtot
        
    def __call__(self, frequency_array, tc=0, phi_c=0):
        in_band = frequency_array <=self.fisco
        hoff = (
            self.amplitude(frequency_array) *
            xp.exp(-1j * self.phase(frequency_array)))
        hoff[~in_band] = 0
        return hoff
    
    def amplitude(self, frequency_array):
        frequency_array = self.mtot * frequency_array
        a_0 = (
          (2 * self.eta / 3 / np.pi**(1 / 3))**0.5 * frequency_array**(-7 / 6))
        amp = 0
        for ii in range(7):
            amp += self.amp_term(ii) * (np.pi * frequency_array)**(ii / 3)
        return amp * a_0 / 3.63082874e+23 / 0.43307112  # kludge amplitude


    def phase(self, frequency_array, tc=0, phi_c=0):
        phi = - phi_c - 4 * np.pi / 3
#         phi = 2 * np.pi * frequency_array * tc - phi_c  # - np.pi / 4
        frequency_array = self.mtot * frequency_array
        outer = 3 / 128 / self.eta * (np.pi * frequency_array)**(-5 / 3)
        for ii in range(8):
            phi += (
                outer * self.phase_term(ii, frequency_array) *
                (np.pi * frequency_array)**(ii / 3))
        return phi

    def phase_term(self, ii, frequency_array):
        if ii == 0:
            return self.phi_0(frequency_array)
        elif ii == 1:
            return self.phi_1(frequency_array)
        elif ii == 2:
            return self.phi_2(frequency_array)
        elif ii == 3:
            return self.phi_3(frequency_array)
        elif ii == 4:
            return self.phi_4(frequency_array)
        elif ii == 5:
            return self.phi_5(frequency_array)
        elif ii == 6:
            return self.phi_6(frequency_array)
        elif ii == 7:
            return self.phi_7(frequency_array)
        
    def amp_term(self, ii):
        if ii == 0:
            return self.amp_0()
        elif ii == 1:
            return self.amp_1()
        elif ii == 2:
            return self.amp_2()
        elif ii == 3:
            return self.amp_3()
        elif ii == 4:
            return self.amp_4()
        elif ii == 5:
            return self.amp_5()
        elif ii == 6:
            return self.amp_6()
    
    def phi_0(self, frequency_array):
        return 1

    def phi_1(self, frequency_array):
        return 0

    def phi_2(self, frequency_array):
        return 3715 / 756 + 55 * self.eta / 9

    def phi_3(self, frequency_array):
        phase = - 16 * np.pi + 113 * self.delta * self.chia / 3
        phase += (113 / 3 - 76 * self.eta / 3) * self.chis
        return phase

    def phi_4(self, frequency_array):
        phase = (15293365 / 508032 + 27145 * self.eta / 504 +
                 3085 * self.eta**2 / 72)
        phase += (-405 / 8 + 200 * self.eta) * self.chia**2
        phase += -405 / 4 * self.delta * self.chia * self.chis
        phase += (-405 / 8 + 5 * self.eta / 2) * self.chis**2
        return phase

    def phi_5(self, frequency_array):
        phase = 38645 * np.pi / 756 - 65 * np.pi * self.eta / 9
        phase += self.delta * (-732985 / 2286 - 140 * self.eta / 9) * self.chia
        phase += (-732985 / 2268 + 24260 * self.eta / 81 +
                  340 * self.eta**2 / 9) * self.chis
        phase *= (1 + xp.log(np.pi * frequency_array))
        return phase

    def phi_6(self, frequency_array):
        phase = 11583231236531 / 4694215680 - 6848 * GAMMA_E / 21
        phase -= 640 * np.pi**2 / 3
        phase += (-15737765635 / 3048192 + 2255 * np.pi**2 / 12) * self.eta
        phase += 76055 * self.eta**2 / 1728
        phase += -127825 * self.eta**3 / 1296
        phase += -6846 / 63 * np.log(64 * np.pi * frequency_array)
        phase += 2270 / 3 * np.pi * self.delta * self.chia
        phase += (2270 * np.pi / 3 - 520 * np.pi * self.eta) * self.chis
        return phase

    def phi_7(self, frequency_array):
        phase = 77096675 * np.pi / 254016
        phase += 378515 * np.pi * self.eta / 1512
        phase -= 74045 * np.pi * self.eta**2 / 756
        phase += self.delta * (-25150083775 / 3048192 +
                               26804935 * self.eta / 6048 -
                               1985 * self.eta**2 / 48) * self.chia
        phase += (-25150083775 / 3048192 + 10566655595 * self.eta / 762048 -
                  1042165 * self.eta**2 / 3024 +
                  5345 * self.eta**3 / 36) * self.chis
        return phase
    
    def amp_0(self):
        return 1

    def amp_1(self):
        return 0

    def amp_2(self):
        return - 323 / 224 + 451 * self.eta / 168

    def amp_3(self):
        amp = 27 * self.delta * self.chia / 8
        amp += (27 / 8 - 11 * self.eta / 6) * self.chis
        return amp

    def amp_4(self):
        amp = -27312085 / 8128512 - 1975055 * self.eta / 338688
        amp += 105271 * self.eta**2 / 24192
        amp += (-81 / 32 + 8 * self.eta) * self.chia**2
        amp += - 81 / 16 * self.delta * self.chia * self.chis
        amp += (-81 / 32 + 17 * self.eta / 8) * self.chis**2
        return amp

    def amp_5(self):
        amp = -85 * np.pi / 64 + 85 * np.pi * self.eta / 16
        amp += self.delta * (285197 / 16128 -
                             1579 * self.eta / 4032) * self.chia
        amp += (285197 / 16128 - 15317 * self.eta / 672
                - 2227 * self.eta**2 / 1008) * self.chis
        return amp

    def amp_6(self):
        amp = -177520268561 / 8583708672
        amp += (545384828789 / 5007163392 - 205 * np.pi**2 / 48) * self.eta
        amp += -3248849057 * self.eta**2 / 178827264
        amp += 34473079 * self.eta**3 / 6386688
        amp += (1614569 / 64512 - 1873643 * self.eta / 16128
                + 2167 * self.eta**2 / 42) * self.chia**2
        amp += (31 * np.pi / 12 - 7 * np.pi * self.eta / 3) * self.chis
        amp += (1614569 / 64512 - 61391 * self.eta / 1344
                + 57451 * self.eta**2 / 4032) * self.chis**2
        amp += self.delta * self.chia * (
            31 * np.pi / 12 + (1614569 / 32256 -
                               165961 * np.pi / 2688) * self.chis)
        return amp
    
    
class TF2_np(TF2):
    
    def __call__(self, frequency_array, tc=0, phi_c=0):
        return (
            self.amplitude(frequency_array) *
            np.exp(-1j * self.phase(frequency_array)))
    
    def phi_5(self, frequency_array):
        phase = 38645 * np.pi / 756 - 65 * np.pi * self.eta / 9
        phase += self.delta * (-732985 / 2286 - 140 * self.eta / 9) * self.chia
        phase += (-732985 / 2268 + 24260 * self.eta / 81 +
                  340 * self.eta**2 / 9) * self.chis
        phase *= (1 + np.log(np.pi * frequency_array))
        return phase


def call_cupy_tf2(frequency_array, mass_1, mass_2, chi_1, chi_2,
                  luminosity_distance, **kwargs):
  
    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    
    in_band = frequency_array >= minimum_frequency
    
    frequency_array = xp.asarray(frequency_array)
    
    h_out_of_band = xp.zeros(int(xp.sum(~in_band)))
    
    wf = TF2(mass_1, mass_2, chi_1, chi_2)
    hplus = wf(frequency_array[in_band])
    hplus = xp.hstack([h_out_of_band, hplus]) / luminosity_distance
    hcross = hplus * xp.exp(-1j * np.pi / 2)
    return dict(plus=hplus, cross=hcross)


class TF2_WFG(object):
    
    def __init__(
            self, duration, sampling_frequency,
            frequency_domain_source_model=call_cupy_tf2,
            waveform_arguments=dict(minimum_frequency=10),
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters):
        self.fdsm = frequency_domain_source_model
        self.waveform_arguments = waveform_arguments
        self.frequency_array = xp.asarray(create_frequency_series(
            duration=duration, sampling_frequency=sampling_frequency))
        self.conversion = parameter_conversion
        
    def frequency_domain_strain(self, parameters):
        parameters, _ = self.conversion(parameters.copy())
        parameters.update(self.waveform_arguments)
        return self.fdsm(self.frequency_array, **parameters)
        
