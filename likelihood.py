import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from bilby.core.likelihood import Likelihood


class CUPYGravitationalWaveTransient(Likelihood):

    def __init__(self, interferometers, waveform_generator, priors=None,
                 distance_marginalization=True):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood. Does
        not include distance/phase marginalization.


        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains the
            detector data and power spectral densities
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
        self._data_to_gpu()
        if priors is None:
            self.priors = priors
        else:
            self.priors = priors.copy()
        self.distance_marginalization = distance_marginalization
        if self.distance_marginalization:
            self._setup_distance_marginalization()
            priors['luminosity_distance'] = priors['luminosity_distance'].minimum
        
    def _data_to_gpu(self):
        for ifo in self.interferometers:
            self.psds[ifo.name] = xp.asarray(
                ifo.power_spectral_density_array[ifo.frequency_mask])
            self.strain[ifo.name] = xp.asarray(
                ifo.frequency_domain_strain[ifo.frequency_mask])
        self.frequency_array = xp.asarray(
            ifo.frequency_array[ifo.frequency_mask])
        self.duration = ifo.strain_data.duration

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={})'\
            .format(self.interferometers, self.waveform_generator)

    def noise_log_likelihood(self):
        """ Calculates the real part of noise log-likelihood

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        if np.isnan(self._noise_log_l):
            log_l = 0
            for interferometer in self.interferometers:
                name = interferometer.name
                log_l -= 2. / self.duration * xp.sum(
                    xp.abs(self.strain[name]) ** 2 / self.psds[name])
            self._noise_log_l = log_l
        return self._noise_log_l

    def log_likelihood_ratio(self):
        """ Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        log_l = 0
        
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            self.parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        d_inner_h = 0
        h_inner_h = 0

        for interferometer in self.interferometers:
            name = interferometer.name
            signal_ifo = xp.sum(xp.vstack([
                waveform_polarizations[mode] * float(
                    interferometer.antenna_response(
                        self.parameters['ra'], self.parameters['dec'],
                        self.parameters['geocent_time'], self.parameters['psi'],
                        mode))
                    for mode in waveform_polarizations]), axis=0)[
                interferometer.frequency_mask]
            
            time_delay = (
                self.parameters['geocent_time'] -
                interferometer.strain_data.start_time +
                interferometer.time_delay_from_geocenter(
                    self.parameters['ra'], self.parameters['dec'],
                    self.parameters['geocent_time']))
            
            signal_ifo *= xp.exp(-2j * time_delay * self.frequency_array)
            
            d_inner_h += xp.sum(xp.conj(signal_ifo) * self.strain[name] /
                                self.psds[name])
            h_inner_h += xp.sum(xp.abs(signal_ifo)**2 / self.psds[name])
            
        if self.distance_marginalization:
            log_l = 0
            log_l += (
                4 / self.duration * d_inner_h *
                self.parameters['luminosity_distance'] / self.distance_array)
            log_l -= (
                2 / self.duration * h_inner_h *
                self.parameters['luminosity_distance']**2 /
                self.distance_array**2)
            log_l = xp.log(xp.sum(xp.exp(log_l) * self.distance_prior_array))
        else:
            log_l = - 2 / self.duration * h_inner_h + 4 / self.duration * d_inner_h
        return float(log_l.real)
    
    def _setup_distance_marginalization(self):
        self.distance_array = np.linspace(
            self.priors['luminosity_distance'].minimum,
            self.priors['luminosity_distance'].maximum, 10000)
        self.distance_prior_array = xp.asarray(
            self.priors['luminosity_distance'].prob(
                self.distance_array)) * (
            self.distance_array[1] - self.distance_array[0])
        self.distance_array = xp.asarray(self.distance_array)
