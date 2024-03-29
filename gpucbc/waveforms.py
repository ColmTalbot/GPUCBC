from collections import namedtuple

import numpy as np
from astropy import constants
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.core.utils import create_frequency_series

try:
    import cupy as xp
except ImportError:
    xp = np

from . import pn

SOLAR_RADIUS_IN_M = constants.GM_sun.si.value / constants.c.si.value ** 2
SOLAR_RADIUS_IN_S = constants.GM_sun.si.value / constants.c.si.value ** 3
MEGA_PARSEC_SI = constants.pc.si.value * 1e6


class Phasing(object):
    def __init__(self, order=16):
        self.v = np.zeros(order, dtype=float)
        self.vlogv = np.zeros(order, dtype=float)
        self.vlogvsq = np.zeros(order, dtype=float)


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

    phase_parameters = namedtuple(
        "PhaseParameters",
        [
            "eta",
            "chi_1",
            "chi_2",
            "m1_on_m",
            "m2_on_m",
            "qm_def_1",
            "qm_def_2",
            "lambda_1",
            "lambda_2",
        ],
    )

    def __init__(
        self, mass_1, mass_2, chi_1, chi_2, luminosity_distance, lambda_1=0, lambda_2=0
    ):
        self._phasing = Phasing()
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.total_mass = mass_1 + mass_2
        self.eta = mass_1 * mass_2 / self.total_mass ** 2

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.quadmon_1 = eos_q_from_lambda(lambda_1)
        self.quadmon_2 = eos_q_from_lambda(lambda_2)

        self.luminosity_distance = luminosity_distance * MEGA_PARSEC_SI

        self.pn_phase_order = -1
        self.pn_spin_order = -1
        self.pn_tidal_order = -1

        self.args = self.phase_parameters(
            self.eta,
            self.chi_1,
            self.chi_2,
            self.mass_1 / self.total_mass,
            self.mass_2 / self.total_mass,
            self.quadmon_1,
            self.quadmon_2,
            self.lambda_1,
            self.lambda_2,
        )

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
        d_energy_d_flux = 5 / 32 / self.eta / orbital_speed ** 9
        amp = amp_0 * d_energy_d_flux ** 0.5 * orbital_speed

        return amp

    def _lal_phasing_coefficients(self):
        from lal import CreateDict
        from lalsimulation import (
            SimInspiralWaveformParamsInsertTidalLambda1,
            SimInspiralWaveformParamsInsertTidalLambda2,
            SimInspiralSetQuadMonParamsFromLambdas,
            SimInspiralTaylorF2AlignedPhasing,
        )

        param_dict = CreateDict()
        SimInspiralWaveformParamsInsertTidalLambda1(param_dict, self.lambda_1)
        SimInspiralWaveformParamsInsertTidalLambda2(param_dict, self.lambda_2)
        SimInspiralSetQuadMonParamsFromLambdas(param_dict)
        return SimInspiralTaylorF2AlignedPhasing(
            self.mass_1, self.mass_2, self.chi_1, self.chi_2, param_dict
        )

    def phasing_coefficients(self):
        scale = 3 / (128 * self.eta)
        self._phasing.v[0] = pn.taylor_f2_phase_0(self.args)
        self._phasing.v[1] = pn.taylor_f2_phase_1(self.args)
        self._phasing.v[2] = pn.taylor_f2_phase_2(self.args)
        self._phasing.v[3] = pn.taylor_f2_phase_3(self.args)
        self._phasing.v[4] = pn.taylor_f2_phase_4(self.args)
        self._phasing.v[5] = pn.taylor_f2_phase_5(self.args)
        self._phasing.v[6] = pn.taylor_f2_phase_6(self.args)
        self._phasing.v[7] = pn.taylor_f2_phase_7(self.args)
        self._phasing.v[10] = pn.taylor_f2_phase_10(self.args)
        self._phasing.v[12] = pn.taylor_f2_phase_12(self.args)
        self._phasing.v[13] = pn.taylor_f2_phase_13(self.args)
        self._phasing.v[14] = pn.taylor_f2_phase_14(self.args)
        if self.pn_tidal_order > 14:
            self._phasing.v[15] = pn.taylor_f2_phase_15(self.args)
        self._phasing.vlogv[5] = pn.taylor_f2_phase_5l(self.args)
        self._phasing.vlogv[6] = pn.taylor_f2_phase_6l(self.args)
        self._phasing.v *= scale
        self._phasing.vlogv *= scale
        self._phasing.vlogvsq *= scale
        return self._phasing

    def phase(self, frequency_array, phi_c=0, orbital_speed=None):
        if orbital_speed is None:
            orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        phase_coefficients = self.phasing_coefficients()
        phasing = xp.zeros(orbital_speed.shape)
        cumulative_power_frequency = orbital_speed ** -5
        log_orbital_speed = xp.log(orbital_speed)
        for ii in range(len(phase_coefficients.v)):
            phasing += phase_coefficients.v[ii] * cumulative_power_frequency
            phasing += (
                phase_coefficients.vlogv[ii]
                * cumulative_power_frequency
                * log_orbital_speed
            )
            cumulative_power_frequency *= orbital_speed

        phasing -= 2 * phi_c + np.pi / 4

        return phasing

    @property
    def pn_tidal_order(self):
        return self._pn_tidal_order

    @pn_tidal_order.setter
    def pn_tidal_order(self, value):
        if value == -1:
            value = 14
        self._pn_tidal_order = value


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
    strain = xp.hstack([h_out_of_band, strain])
    h_plus = strain * (1 + np.cos(theta_jn) ** 2) / 2
    h_cross = -1j * strain * np.cos(theta_jn)

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


def eos_q_from_lambda(lamb, tolerance=0.5):
    """
    A translation of lalsimulation.SimInspiralEOSQfromLambda

    Calculates the quadrupole monopole term using universal relations.
    """

    def worker(log_lambda):
        return np.exp(
            0.194
            + 0.0936 * log_lambda
            + 0.0474 * log_lambda ** 2
            - 0.00421 * log_lambda ** 3
            + 0.000123 * log_lambda ** 4
        )

    if isinstance(lamb, (float, int)):
        if lamb < tolerance:
            return 1
        else:
            log_lambda = np.log(lamb)
            return worker(log_lambda)
    else:
        lamb = np.array(lamb, dtype=float)
        quadmon = np.ones(lamb.shape)
        _mask = lamb >= tolerance
        log_lambda = np.log(lamb[_mask])
        quadmon[_mask] = worker(log_lambda)
    return quadmon
