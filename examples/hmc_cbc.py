import sys
from functools import partial

import bilby
import gpucbc
gpucbc.activate_jax()  # noqa
import numpy as np
from jax import numpy as jnp

from utils import (
    generate_cbc,
    generate_waveform,
    ln_likelihood,
    plot_corner,
    run_sampler,
    set_data,
    unit_normal_ln_pdf,
)

duration = int(sys.argv[1])

priors = bilby.core.prior.PriorDict()
priors["chirp_mass"] = bilby.core.prior.Uniform(30, 50)
# priors["mass_ratio"] = bilby.core.prior.Uniform(0, 2)
priors["mass_ratio"] = bilby.core.prior.Normal(0, 1)
priors["chi_plus"] = bilby.core.prior.Uniform(-1, 1)
priors["chi_minus"] = bilby.core.prior.Uniform(-1, 1)
priors["luminosity_distance"] = bilby.core.prior.Uniform(20, 6000)
priors["delta_time"] = bilby.core.prior.Uniform(0, 1)
# priors["phase"] = bilby.core.prior.Uniform(0, np.pi)
priors["phase_x"] = bilby.core.prior.Normal(0, 1)
priors["phase_y"] = bilby.core.prior.Normal(0, 1)
# priors["cos_theta_jn"] = bilby.core.prior.Uniform(-1, 1)
priors["cos_theta_jn"] = bilby.core.prior.Normal(0, 1)
priors["cos_zenith"] = bilby.core.prior.Uniform(-1, 1)
priors["azimuth"] = bilby.core.prior.Uniform(0, 2 * np.pi)
# priors["psi"] = bilby.core.prior.Uniform(0, np.pi / 2)
priors["psi_x"] = bilby.core.prior.Normal(0, 1)
priors["psi_y"] = bilby.core.prior.Normal(0, 1)
ndim = len(priors)
search_keys = list(priors.keys())
labels = [key.replace("_", " ") for key in priors]
minima = jnp.asarray([priors[key].minimum for key in search_keys])
maxima = jnp.asarray([priors[key].maximum for key in search_keys])
domain = maxima - minima
# true_parameters = jnp.asarray(list(priors.sample().values()))
true_parameters = jnp.asarray(
    [40, 1, 0.3, -0.2, 2000, 0.5, 0.5, -0.5, 3, -0.5, np.pi, 0.6, -0.3]
)
start = true_parameters

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1", "K1"][:2])
fmin = 20
fmax = 100
for ifo in ifos:
    ifo.minimum_frequency = fmin
    ifo.maximum_frequency = fmax
# ifos.set_strain_data_from_power_spectral_densities(
ifos.set_strain_data_from_zero_noise(
    duration=duration,
    sampling_frequency=2048,
)
frequencies = jnp.arange(fmin, fmax + 1 / duration, 1 / duration)
generator = partial(generate_waveform, wfg=generate_cbc, frequencies=frequencies)
data, asd = set_data(ifos, generator, true_parameters)

_ln_likelihood_fn = partial(
    ln_likelihood, wfg=generator, data=data, asd=asd, duration=duration,
)


def ln_prior_fn(mu):
    ln_prior = jnp.sum(jnp.log((mu > minima) * (mu < maxima)))
    ln_prior += unit_normal_ln_pdf(mu[1])
    ln_prior += jnp.sum(jnp.log(-jnp.log(jnp.abs(mu[2:4])) / 2))
    ln_prior += jnp.sum(unit_normal_ln_pdf(mu[-7:-4]))
    ln_prior += jnp.sum(unit_normal_ln_pdf(mu[-2:]))
    return jnp.nan_to_num(ln_prior, nan=-jnp.inf)


print(_ln_likelihood_fn(start), ln_prior_fn(start))
mcmc = run_sampler(
    start=start,
    num_warmup=1000,
    num_samples=5000,
    ln_prior_fn=ln_prior_fn,
    ln_likelihood_fn=_ln_likelihood_fn,
)
plot_corner(mcmc, labels)

import IPython

IPython.embed()
