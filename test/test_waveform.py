#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd

import lal
import lalsimulation
from bilby.core.prior import Constraint, Uniform, Cosine, Sine, PriorDict
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.gw.detector import get_empty_interferometer
from bilby.gw.prior import BNSPriorDict, UniformSourceFrame, AlignedSpin
from bilby.gw.source import lal_binary_neutron_star
from bilby.gw.utils import noise_weighted_inner_product
from bilby.gw.waveform_generator import WaveformGenerator

from gpucbc.waveforms import TF2WFG, TF2


class TF2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.waveform_arguments = dict(
            waveform_approximant="TaylorF2",
            minimum_frequency=20,
            maximum_frequency=1024,
            reference_frequency=20,
            pn_tidal_order=-1,
        )

        self.duration = 4
        self.sampling_frequency = 2048

        self.ifo = get_empty_interferometer("H1")
        self.ifo.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )

        self.bilby_wfg = WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=lal_binary_neutron_star,
            waveform_arguments=self.waveform_arguments,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        )

        self.gpu_wfg = TF2WFG(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            waveform_arguments=self.waveform_arguments,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        )

    def test_native_phasing(self):
        priors = PriorDict()
        priors["mass_1"] = Uniform(1, 100)
        priors["mass_2"] = Uniform(1, 100)
        priors["chi_1"] = AlignedSpin(
            name="chi_1", a_prior=Uniform(minimum=0, maximum=1)
        )
        priors["chi_2"] = AlignedSpin(
            name="chi_2", a_prior=Uniform(minimum=0, maximum=1)
        )
        priors["lambda_1"] = Uniform(name="lambda_1", minimum=0, maximum=1000)
        priors["lambda_2"] = Uniform(name="lambda_2", minimum=0, maximum=1000)
        priors["luminosity_distance"] = Uniform(10, 200)

        wf = TF2(**priors.sample())
        lal_phasing = wf._lal_phasing_coefficients()
        my_phasing = wf.phasing_coefficients()
        self.assertLess(max(abs(lal_phasing.v - my_phasing.v)), 1e-8)
        self.assertLess(max(abs(lal_phasing.vlogv - my_phasing.vlogv)), 1e-8)
        self.assertLess(max(abs(lal_phasing.vlogvsq - my_phasing.vlogvsq)), 1e-8)

    def test_absolute_overlap(self):
        np.random.seed(42)
        priors = BNSPriorDict(aligned_spin=True)
        priors["mass_1"] = Uniform(1, 100)
        priors["mass_2"] = Uniform(1, 100)
        # priors["total_mass"] = Uniform(2, 100)
        # priors["mass_ratio"] = Uniform(name="mass_ratio", minimum=0.5, maximum=1)
        # priors["mass_1"] = Constraint(name="mass_1", minimum=1, maximum=50)
        # priors["mass_2"] = Constraint(name="mass_2", minimum=1, maximum=50)
        del priors["mass_ratio"], priors["chirp_mass"]
        priors["luminosity_distance"] = UniformSourceFrame(
            name="luminosity_distance", minimum=1e2, maximum=5e3
        )
        priors["dec"] = Cosine(name="dec")
        priors["ra"] = Uniform(
            name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
        )
        priors["theta_jn"] = Sine(name="theta_jn")
        priors["psi"] = Uniform(
            name="psi", minimum=0, maximum=np.pi, boundary="periodic"
        )
        priors["phase"] = Uniform(
            name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
        )
        priors["chi_1"] = AlignedSpin(
            name="chi_1", a_prior=Uniform(minimum=0, maximum=1)
        )
        priors["chi_2"] = AlignedSpin(
            name="chi_2", a_prior=Uniform(minimum=0, maximum=1)
        )
        priors["lambda_1"] = Uniform(name="lambda_1", minimum=0, maximum=5000)
        priors["lambda_2"] = Uniform(name="lambda_2", minimum=0, maximum=5000)
        priors["geocent_time"] = 0
        # priors["geocent_time"] = Uniform(-10, 10)

        n_samples = 100
        all_parameters = pd.DataFrame(priors.sample(n_samples))
        overlaps = list()
        frequencies = np.linspace(0, 1024, 1024 * 4 + 1)

        for ii in range(n_samples):
            parameters = dict(all_parameters.iloc[ii])
            params = parameters

            ldict = lal.CreateDict()
            lalsimulation.SimInspiralWaveformParamsInsertPNTidalOrder(ldict, -1)
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(ldict, params["lambda_1"])
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(ldict, params["lambda_2"])
            lal_strain = lalsimulation.SimInspiralChooseFDWaveform(
                params["mass_1"] * lal.MSUN_SI,
                params["mass_2"] * lal.MSUN_SI,
                0.0,
                0.0,
                params["chi_1"],
                0.0,
                0.0,
                params["chi_2"],
                params["luminosity_distance"] * 1e6 * lal.PC_SI,
                params["theta_jn"],
                params["phase"],
                0.0,
                0.0,
                0.0,
                1 / self.duration,
                20.0,
                1024.0,
                0.0,
                ldict,
                lalsimulation.TaylorF2,
            )
            bilby_pols = dict(plus=lal_strain[0].data.data, cross=lal_strain[1].data.data)
            gpu_pols = self.gpu_wfg.frequency_domain_strain(parameters)

            bilby_strain = self.ifo.get_detector_response(
                waveform_polarizations=bilby_pols, parameters=parameters
            )

            gpu_strain = self.ifo.get_detector_response(
                waveform_polarizations=gpu_pols, parameters=parameters
            )

            inner_product = noise_weighted_inner_product(
                aa=bilby_strain,
                bb=gpu_strain,
                power_spectral_density=self.ifo.power_spectral_density_array,
                duration=self.duration,
            )
            overlap = (
                inner_product
                / self.ifo.optimal_snr_squared(signal=bilby_strain) ** 0.5
                / self.ifo.optimal_snr_squared(signal=gpu_strain) ** 0.5
            )
            # print(self.ifo.optimal_snr_squared(signal=bilby_strain) ** 0.5, self.ifo.optimal_snr_squared(signal=gpu_strain) ** 0.5)
            # print(np.max(abs(
            #     np.fft.fft(bilby_strain[:-1] * gpu_strain.conj().real[:-1]) / self.ifo.power_spectral_density_array[:-1]
            #     / self.ifo.optimal_snr_squared(signal=bilby_strain) ** 0.5
            #     / self.ifo.optimal_snr_squared(signal=gpu_strain) ** 0.5
            # )))
            overlaps.append(overlap)
        print(overlaps)
        self.assertTrue(min(np.abs(overlaps)) > 0.99)
