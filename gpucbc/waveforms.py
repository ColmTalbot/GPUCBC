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

        self.delta = (self.mass_1 - self.mass_2) / self.total_mass
        self.chis = (self.chi_1 + self.chi_2) / 2
        self.chia = (self.chi_1 - self.chi_2) / 2
        self.chi_pn = (
                              self.mass_1 * self.chi_1 + self.mass_2 * self.chi_2
        ) / self.total_mass - 38 * self.symmetric_mass_ratio / 113 * (
            self.chi_1 + self.chi_2
        )
        seta = np.sqrt(1.0 - 4.0 * self.symmetric_mass_ratio)
        mass_1 = 0.5 * (1.0 + seta)
        mass_2 = 0.5 * (1.0 - seta)
        self.chi = mass_1 ** 2 * self.chi_1 + mass_2 ** 2 * self.chi_2
        self.fisco = 6 ** (-1.5) / np.pi / self.total_mass
        self.luminosity_distance = luminosity_distance / MEGA_PARSEC_SI

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

    def phase_term(self, ii, frequency_array):
        if ii == 0:
            phase = self.phi_0(frequency_array)
        elif ii == 1:
            phase = self.phi_1(frequency_array)
        elif ii == 2:
            phase = self.phi_2(frequency_array)
        elif ii == 3:
            phase = self.phi_3(frequency_array)
        elif ii == 4:
            phase = self.phi_4(frequency_array)
        elif ii == 5:
            phase = self.phi_5(frequency_array)
        elif ii == 6:
            phase = self.phi_6(frequency_array)
        elif ii == 7:
            phase = self.phi_7(frequency_array)
        else:
            phase = 0
        return phase

    def amp_term(self, ii):
        if ii == 0:
            amplitude = self.amp_0()
        elif ii == 1:
            amplitude = self.amp_1()
        elif ii == 2:
            amplitude = self.amp_2()
        elif ii == 3:
            amplitude = self.amp_3()
        elif ii == 4:
            amplitude = self.amp_4()
        elif ii == 5:
            amplitude = self.amp_5()
        elif ii == 6:
            amplitude = self.amp_6()
        else:
            amplitude = 0
        return amplitude

    def phi_0(self, frequency_array):
        return 1

    def phi_1(self, frequency_array):
        return 0

    def phi_2(self, frequency_array):
        return 3715 / 756 + 55 * self.symmetric_mass_ratio / 9

    def phi_3(self, frequency_array):
        phase = -16 * np.pi + 113 * self.delta * self.chia / 3
        phase += (113 / 3 - 76 * self.symmetric_mass_ratio / 3) * self.chis
        return phase

    def phi_4(self, frequency_array):
        phase = (
            15293365 / 508032
            + 27145 * self.symmetric_mass_ratio / 504
            + 3085 * self.symmetric_mass_ratio ** 2 / 72
        )
        phase += (-405 / 8 + 200 * self.symmetric_mass_ratio) * self.chia ** 2
        phase += -405 / 4 * self.delta * self.chia * self.chis
        phase += (-405 / 8 + 5 * self.symmetric_mass_ratio / 2) * self.chis ** 2
        return phase

    def phi_5(self, frequency_array):
        phase = 38645 * np.pi / 756 - 65 * np.pi * self.symmetric_mass_ratio / 9
        phase += (
            self.delta
            * (-732985 / 2286 - 140 * self.symmetric_mass_ratio / 9)
            * self.chia
        )
        phase += (
            -732985 / 2268
            + 24260 * self.symmetric_mass_ratio / 81
            + 340 * self.symmetric_mass_ratio ** 2 / 9
        ) * self.chis
        phase *= 1 + xp.log(np.pi * frequency_array)
        return phase

    def phi_6(self, frequency_array):
        phase = 11583231236531 / 4694215680 - 6848 * np.euler_gamma / 21
        phase -= 640 * np.pi ** 2 / 3
        phase += (
            -15737765635 / 3048192 + 2255 * np.pi ** 2 / 12
        ) * self.symmetric_mass_ratio
        phase += 76055 * self.symmetric_mass_ratio ** 2 / 1728
        phase += -127825 * self.symmetric_mass_ratio ** 3 / 1296
        phase += -6846 / 63 * np.log(64 * np.pi * frequency_array)
        phase += 2270 / 3 * np.pi * self.delta * self.chia
        phase += (
            2270 * np.pi / 3 - 520 * np.pi * self.symmetric_mass_ratio
        ) * self.chis
        return phase

    def phi_7(self, frequency_array):
        phase = 77096675 * np.pi / 254016
        phase += 378515 * np.pi * self.symmetric_mass_ratio / 1512
        phase -= 74045 * np.pi * self.symmetric_mass_ratio ** 2 / 756
        phase += (
            self.delta
            * (
                -25150083775 / 3048192
                + 26804935 * self.symmetric_mass_ratio / 6048
                - 1985 * self.symmetric_mass_ratio ** 2 / 48
            )
            * self.chia
        )
        phase += (
            -25150083775 / 3048192
            + 10566655595 * self.symmetric_mass_ratio / 762048
            - 1042165 * self.symmetric_mass_ratio ** 2 / 3024
            + 5345 * self.symmetric_mass_ratio ** 3 / 36
        ) * self.chis
        return phase

    def amp_0(self):
        return 1

    def amp_1(self):
        return 0

    def amp_2(self):
        return -323 / 224 + 451 * self.symmetric_mass_ratio / 168

    def amp_3(self):
        amp = 27 * self.delta * self.chia / 8
        amp += (27 / 8 - 11 * self.symmetric_mass_ratio / 6) * self.chis
        return amp

    def amp_4(self):
        amp = -27312085 / 8128512 - 1975055 * self.symmetric_mass_ratio / 338688
        amp += 105271 * self.symmetric_mass_ratio ** 2 / 24192
        amp += (-81 / 32 + 8 * self.symmetric_mass_ratio) * self.chia ** 2
        amp += -81 / 16 * self.delta * self.chia * self.chis
        amp += (-81 / 32 + 17 * self.symmetric_mass_ratio / 8) * self.chis ** 2
        return amp

    def amp_5(self):
        amp = -85 * np.pi / 64 + 85 * np.pi * self.symmetric_mass_ratio / 16
        amp += (
            self.delta
            * (285197 / 16128 - 1579 * self.symmetric_mass_ratio / 4032)
            * self.chia
        )
        amp += (
            285197 / 16128
            - 15317 * self.symmetric_mass_ratio / 672
            - 2227 * self.symmetric_mass_ratio ** 2 / 1008
        ) * self.chis
        return amp

    def amp_6(self):
        amp = -177520268561 / 8583708672
        amp += (
            545384828789 / 5007163392 - 205 * np.pi ** 2 / 48
        ) * self.symmetric_mass_ratio
        amp += -3248849057 * self.symmetric_mass_ratio ** 2 / 178827264
        amp += 34473079 * self.symmetric_mass_ratio ** 3 / 6386688
        amp += (
            1614569 / 64512
            - 1873643 * self.symmetric_mass_ratio / 16128
            + 2167 * self.symmetric_mass_ratio ** 2 / 42
        ) * self.chia ** 2
        amp += (31 * np.pi / 12 - 7 * np.pi * self.symmetric_mass_ratio / 3) * self.chis
        amp += (
            1614569 / 64512
            - 61391 * self.symmetric_mass_ratio / 1344
            + 57451 * self.symmetric_mass_ratio ** 2 / 4032
        ) * self.chis ** 2
        amp += (
            self.delta
            * self.chia
            * (31 * np.pi / 12 + (1614569 / 32256 - 165961 * np.pi / 2688) * self.chis)
        )
        return amp


class TF2_np(TF2):
    def __call__(self, frequency_array, tc=0, phi_c=0):
        return self.amplitude(frequency_array) * np.exp(
            -1j * self.phase(frequency_array)
        )

    def phi_5(self, frequency_array):
        phase = 38645 * np.pi / 756 - 65 * np.pi * self.symmetric_mass_ratio / 9
        phase += (
            self.delta
            * (-732985 / 2286 - 140 * self.symmetric_mass_ratio / 9)
            * self.chia
        )
        phase += (
            -732985 / 2268
            + 24260 * self.symmetric_mass_ratio / 81
            + 340 * self.symmetric_mass_ratio ** 2 / 9
        ) * self.chis
        phase *= 1 + np.log(np.pi * frequency_array)
        return phase


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
