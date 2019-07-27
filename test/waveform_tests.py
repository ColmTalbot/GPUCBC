#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd

from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import get_empty_interferometer
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.utils import noise_weighted_inner_product
from bilby.gw.waveform_generator import WaveformGenerator

from gpucbc.waveforms import TF2WFG


class TF2Test(unittest.TestCase):

    def setUp(self) -> None:
        self.waveform_arguments = dict(
            waveform_approximant="TaylorF2",
            minimum_frequency=20,
            maximum_frequency=1024,
            reference_frequency=20,
        )

        self.duration = 4
        self.sampling_frequency = 2048

        self.ifo = get_empty_interferometer("H1")
        self.ifo.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration)

        self.bilby_wfg = WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=lal_binary_black_hole,
            waveform_arguments=self.waveform_arguments,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        )

        self.gpu_wfg = TF2WFG(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            waveform_arguments=self.waveform_arguments,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        )

    def test_absolute_overlap(self):
        priors = BBHPriorDict(aligned_spin=True)
        del priors["mass_1"], priors["mass_2"]
        priors["total_mass"] = Uniform(5, 50)
        priors["mass_ratio"] = Uniform(0.5, 1)
        priors["geocent_time"] = Uniform(-10, 10)

        n_samples = 100
        all_parameters = pd.DataFrame(priors.sample(n_samples))
        overlaps = list()

        for ii in range(n_samples):
            parameters = dict(all_parameters.iloc[ii])
            bilby_pols = self.bilby_wfg.frequency_domain_strain(parameters)
            gpu_pols = self.gpu_wfg.frequency_domain_strain(parameters)
            bilby_strain = self.ifo.get_detector_response(
                waveform_polarizations=bilby_pols, parameters=parameters)
            gpu_strain = self.ifo.get_detector_response(
                waveform_polarizations=gpu_pols, parameters=parameters)
            inner_product = noise_weighted_inner_product(
                aa=bilby_strain, bb=gpu_strain,
                power_spectral_density=self.ifo.power_spectral_density_array,
                duration=self.duration)
            overlap = (
                inner_product /
                self.ifo.optimal_snr_squared(signal=bilby_strain) ** 0.5 /
                self.ifo.optimal_snr_squared(signal=gpu_strain) ** 0.5
            )
            overlaps.append(overlap)
        self.assertTrue(min(np.abs(overlaps)) > 0.995)
