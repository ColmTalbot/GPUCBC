import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from astropy import constants

from lal import CreateDict
import lalsimulation as ls
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.core.utils import create_frequency_series

SOLAR_RADIUS_IN_M = constants.GM_sun.si.value / constants.c.si.value ** 2
SOLAR_RADIUS_IN_S = constants.GM_sun.si.value / constants.c.si.value ** 3
MEGA_PARSEC_SI = constants.pc.si.value * 1e6


class TF2(object):
    """
    A copy of the TaylorF2 waveform.

    Based on the implementation in
    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralTaylorF2.c

    This has not been rigorously tested.

    Parameters
    ----------
    mass_1: float
        Mass of the more massive object in solar masses.
    mass_2: float
        Mass of the less massive object in solar masses.
    chi_1: float
        Dimensionless aligned spin of the more massive object.
    chi_2: float
        Dimensionless aligned spin of the less massive object.
    luminosity_distance: float
        Distance to the binary in Mpc.
    lambda_1: float
        Dimensionless tidal deformability of the more massive object.
    lambda_2: float
        Dimensionless tidal deformability of the less massive object.
    """

    def __init__(
        self, mass_1, mass_2, chi_1, chi_2, luminosity_distance, lambda_1=0, lambda_2=0
    ):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.total_mass = mass_1 + mass_2
        self.symmetric_mass_ratio = mass_1 * mass_2 / self.total_mass ** 2

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.param_dict = CreateDict()
        ls.SimInspiralWaveformParamsInsertTidalLambda1(self.param_dict, self.lambda_1)
        ls.SimInspiralWaveformParamsInsertTidalLambda2(self.param_dict, self.lambda_2)
        ls.SimInspiralSetQuadMonParamsFromLambdas(self.param_dict)

        # self.delta = (self.mass_1 - self.mass_2) / self.total_mass
        # self.chis = (self.chi_1 + self.chi_2) / 2
        # self.chia = (self.chi_1 - self.chi_2) / 2
        # self.chi_pn = (
        #     self.mass_1 * self.chi_1 + self.mass_2 * self.chi_2
        # ) / self.total_mass - 38 * self.symmetric_mass_ratio / 113 * (
        #     self.chi_1 + self.chi_2
        # )
        # seta = np.sqrt(1.0 - 4.0 * self.symmetric_mass_ratio)
        # mass_1 = 0.5 * (1.0 + seta)
        # mass_2 = 0.5 * (1.0 - seta)
        # self.chi = mass_1 ** 2 * self.chi_1 + mass_2 ** 2 * self.chi_2
        # self.fisco = 6 ** (-1.5) / np.pi / self.total_mass
        self.luminosity_distance = luminosity_distance * MEGA_PARSEC_SI

    def __call__(self, frequency_array, tc=0, phi_c=0):
        orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        hoff = self.amplitude(frequency_array, orbital_speed=orbital_speed) * xp.exp(
            -1j * self.phase(frequency_array, phi_c=phi_c, orbital_speed=orbital_speed)
        )
        return hoff

    def orbital_speed(self, frequency_array):
        return (np.pi * self.total_mass * SOLAR_RADIUS_IN_S * frequency_array) ** (
            1 / 3
        )

    def amplitude(self, frequency_array, orbital_speed=None):
        if orbital_speed is None:
            orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        amp_0 = (
            -4
            * self.mass_1
            * self.mass_2
            / self.luminosity_distance
            * SOLAR_RADIUS_IN_M
            * SOLAR_RADIUS_IN_S
            * (np.pi / 12) ** 0.5
        )
        d_energy_d_flux = 5 / 32 / self.symmetric_mass_ratio / orbital_speed ** 9
        amp = amp_0 * d_energy_d_flux ** 0.5 * orbital_speed

        return amp

    def phase(self, frequency_array, phi_c=0, orbital_speed=None):
        if orbital_speed is None:
            orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        phase_coefficients = ls.SimInspiralTaylorF2AlignedPhasing(
            self.mass_1, self.mass_2, self.chi_1, self.chi_2, self.param_dict
        )
        phasing = xp.zeros_like(orbital_speed)
        cumulative_power_frequency = orbital_speed ** -5
        for ii in range(len(phase_coefficients.v)):
            phasing += phase_coefficients.v[ii] * cumulative_power_frequency
            phasing += (
                phase_coefficients.vlogv[ii]
                * cumulative_power_frequency
                * xp.log(orbital_speed)
            )
            cumulative_power_frequency *= orbital_speed

        phasing -= 2 * phi_c + np.pi / 4

        return phasing


class TF2_np(TF2):
    def __call__(self, frequency_array, tc=0, phi_c=0):
        return self.amplitude(frequency_array) * np.exp(
            -1j * self.phase(frequency_array)
        )


def call_cupy_tf2(
    frequency_array,
    mass_1,
    mass_2,
    chi_1,
    chi_2,
    luminosity_distance,
    theta_jn,
    phase,
    lambda_1=0,
    lambda_2=0,
    **kwargs
):

    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    minimum_frequency = waveform_kwargs["minimum_frequency"]

    in_band = frequency_array >= minimum_frequency

    frequency_array = xp.asarray(frequency_array)

    h_out_of_band = xp.zeros(int(xp.sum(~in_band)))

    wf = TF2(
        mass_1,
        mass_2,
        chi_1,
        chi_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        luminosity_distance=luminosity_distance,
    )
    strain = wf(frequency_array[in_band], phi_c=phase)
    h_plus = xp.hstack([h_out_of_band, strain]) * (1 + np.cos(theta_jn) ** 2) / 2
    h_cross = (
        xp.hstack([h_out_of_band, strain]) * xp.exp(-1j * np.pi / 2) * np.cos(theta_jn)
    )

    return dict(plus=h_plus, cross=h_cross)


class TF2WFG(object):
    def __init__(
        self,
        duration,
        sampling_frequency,
        frequency_domain_source_model=call_cupy_tf2,
        waveform_arguments=None,
        parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
    ):
        if waveform_arguments is None:
            waveform_arguments = dict(minimum_frequency=10)
        self.fdsm = frequency_domain_source_model
        self.waveform_arguments = waveform_arguments
        self.frequency_array = xp.asarray(
            create_frequency_series(
                duration=duration, sampling_frequency=sampling_frequency
            )
        )
        self.conversion = parameter_conversion

    def frequency_domain_strain(self, parameters):
        parameters, _ = self.conversion(parameters.copy())
        parameters.update(self.waveform_arguments)
        return self.fdsm(self.frequency_array, **parameters)
