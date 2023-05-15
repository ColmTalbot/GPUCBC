import numpy as np
from bilby.core.likelihood import Likelihood

from .backend import BACKEND as B


class CUPYGravitationalWaveTransient(Likelihood):
    def __init__(
        self,
        interferometers,
        waveform_generator,
        priors=None,
        distance_marginalization=True,
        phase_marginalization=True,
        time_marginalization=False,
    ):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood.
        Does not include time/phase marginalization.


        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains
            the detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters

        """
        Likelihood.__init__(self, dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self._noise_log_l = np.nan
        self.psds = dict()
        self.strain = dict()
        self.frequency_array = dict()
        self._data_to_gpu()
        if priors is None:
            self.priors = priors
        else:
            self.priors = priors.copy()
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        if self.distance_marginalization:
            self._setup_distance_marginalization()
            priors["luminosity_distance"] = priors["luminosity_distance"].minimum
        if self.phase_marginalization:
            priors["phase"] = 0.0
        self.time_marginalization = False

    def _data_to_gpu(self):
        for ifo in self.interferometers:
            self.psds[ifo.name] = B.np.asarray(
                ifo.power_spectral_density_array[ifo.frequency_mask]
            )
            self.strain[ifo.name] = B.np.asarray(
                ifo.frequency_domain_strain[ifo.frequency_mask]
            )
            self.frequency_array[ifo.name] = B.np.asarray(
                ifo.frequency_array[ifo.frequency_mask]
            )
        self.duration = ifo.strain_data.duration

    def noise_log_likelihood(self):
        """Calculates the real part of noise log-likelihood

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        if np.isnan(self._noise_log_l):
            log_l = 0
            for interferometer in self.interferometers:
                name = interferometer.name
                log_l -= (
                    2.0
                    / self.duration
                    * (B.np.abs(self.strain[name]) ** 2 / self.psds[name]).sum()
                )
            self._noise_log_l = float(log_l)
        return self._noise_log_l

    def log_likelihood_ratio(self):
        """Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            self.parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        d_inner_h = 0
        h_inner_h = 0

        for interferometer in self.interferometers:
            d_inner_h_ifo, h_inner_h_ifo = self.calculate_snrs(
                interferometer=interferometer,
                waveform_polarizations=waveform_polarizations,
            )
            d_inner_h += d_inner_h_ifo
            h_inner_h += h_inner_h_ifo

        if self.distance_marginalization:
            log_l = self.distance_marglinalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=h_inner_h
            )
        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=h_inner_h
            )
        else:
            log_l = - h_inner_h / 2 + d_inner_h
        return float(log_l.real)

    def calculate_snrs(self, interferometer, waveform_polarizations):
        name = interferometer.name
        signal_ifo = B.np.vstack(
            [
                waveform_polarizations[mode]
                * float(
                    interferometer.antenna_response(
                        self.parameters["ra"],
                        self.parameters["dec"],
                        self.parameters["geocent_time"],
                        self.parameters["psi"],
                        mode,
                    )
                )
                for mode in waveform_polarizations
            ]
        ).sum(axis=0)[interferometer.frequency_mask]

        time_delay = (
            self.parameters["geocent_time"]
            - interferometer.strain_data.start_time
            + interferometer.time_delay_from_geocenter(
                self.parameters["ra"],
                self.parameters["dec"],
                self.parameters["geocent_time"],
            )
        )

        signal_ifo *= B.np.exp(-2j * np.pi * time_delay * self.frequency_array[name])

        d_inner_h = (signal_ifo.conj() * self.strain[name] / self.psds[name]).sum()
        h_inner_h = (B.np.abs(signal_ifo) ** 2 / self.psds[name]).sum()
        d_inner_h *= 4 / self.duration
        h_inner_h *= 4 / self.duration
        return d_inner_h, h_inner_h

    def distance_marglinalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h_array = (
            d_inner_h
            * self.parameters["luminosity_distance"]
            / self.distance_array
        )
        h_inner_h_array = (
            h_inner_h
            * self.parameters["luminosity_distance"] ** 2
            / self.distance_array ** 2
        )
        if self.phase_marginalization:
            log_l_array = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h_array, h_inner_h=h_inner_h_array
            )
        else:
            log_l_array = - h_inner_h_array / 2 + d_inner_h_array.real
        log_l = logsumexp(log_l_array, b=self.distance_prior_array)
        return log_l

    def phase_marginalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h = B.np.abs(d_inner_h)
        d_inner_h = B.np.log(B.special.i0e(d_inner_h)) + d_inner_h
        log_l = - h_inner_h / 2 + d_inner_h
        return log_l

    def _setup_distance_marginalization(self):
        self.distance_array = np.linspace(
            self.priors["luminosity_distance"].minimum,
            self.priors["luminosity_distance"].maximum,
            10000,
        )
        self.distance_prior_array = B.np.asarray(
            self.priors["luminosity_distance"].prob(self.distance_array)
        ) * (self.distance_array[1] - self.distance_array[0])
        self.distance_array = B.np.asarray(self.distance_array)

    def generate_posterior_sample_from_marginalized_likelihood(self):
        return self.parameters.copy()
