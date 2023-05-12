import sys
from functools import partial

import bilby
import gpucbc
gpucbc.activate_jax()  # noqa
from gpucbc.waveforms import TF2
import arviz
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import NUTS, MCMC

import utils
from utils import (
    generate_cbc,
    generate_waveform,
    extrinsic_ln_prior,
    extrinsic_priors,
    ln_likelihood,
    plot_corner,
    run_sampler,
    set_data,
    unit_normal_ln_pdf,
)
from model import model

duration = int(sys.argv[1])

priors = bilby.core.prior.PriorDict()
priors["chirp_mass"] = bilby.core.prior.Uniform(30, 40)
# priors["mass_ratio"] = bilby.core.prior.Uniform(0, 2)
priors["mass_ratio"] = bilby.core.prior.Normal(0, 1)
priors["chi_eff"] = bilby.core.prior.Normal(0, 0.1)
# priors["chi_1"] = bilby.gw.prior.AlignedSpin()
# priors["chi_2"] = bilby.gw.prior.AlignedSpin()
priors["luminosity_distance"] = bilby.core.prior.Uniform(0, 2)
priors.update(extrinsic_priors())
ndim = len(priors)
search_keys = list(priors.keys())
labels = [key.replace("_", " ") for key in priors]
# true_parameters = jnp.asarray(list(priors.sample().values()))
true_parameters = jnp.asarray([
    35, 0.8, 0.0, 1, -0.01, 0.2, 0.4, 0.3, 0.9, 0.5,
])
# true_parameters = jnp.array([ 1.58035168,  0.0061823 ,  0.51604031,  0.23270815,
#               0.18904092,  0.37383563,  1.58129625,  0.98758333,
#               1.54886385,  1.10747666, -1.71131634,  1.81865602,
#              -1.10737442])
# priors["chirp_mass"].minimum = 0.1
# priors["chirp_mass"].maximum = 100
priors["chirp_mass"].minimum = float(true_parameters[0] - 0.01)
priors["chirp_mass"].maximum = float(true_parameters[0] + 0.01)
minima = jnp.asarray([priors[key].minimum for key in search_keys])
maxima = jnp.asarray([priors[key].maximum for key in search_keys])
domain = maxima - minima
utils.domain = domain
# true_parameters = jnp.asarray(
#     [40, 1, 0.3, -0.2, 1, 0.5, 0.5, -0.5, 3, -0.5, 0.4, 1.2, 0.6]
# )
start = true_parameters

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1", "K1"][:3])
fmin = 20
fmax = 100
utils.REFERENCE_FREQUENCY = 60
for ifo in ifos:
    ifo.minimum_frequency = fmin
    ifo.maximum_frequency = fmax
# ifos.set_strain_data_from_power_spectral_densities(
ifos.set_strain_data_from_zero_noise(
    duration=duration,
    sampling_frequency=2048,
)
frequencies = jnp.arange(fmin, fmax + 1 / duration, 1 / duration)
TF2.update_cache(frequency_array=frequencies)
generator = partial(generate_waveform, wfg=generate_cbc, frequencies=frequencies)
data, asd = set_data(ifos, generator, true_parameters)

kernel = NUTS(
    model,
    adapt_mass_matrix=True,
    dense_mass=True,
)
mcmc = MCMC(kernel, num_warmup=200, num_samples=1000)
mcmc.run(
    PRNGKey(np.random.randint(0, 100000)),
    generator, data, asd, duration,
    init_params=start,
    beta=1,
)
mcmc.print_summary(exclude_deterministic=False)
azdata = arviz.from_numpyro(mcmc)
arviz.plot_trace(azdata)
plt.tight_layout()
plt.savefig("trace.png")
plt.close()
arviz.plot_pair(
    azdata,
    marginals=True,
    kind="hexbin",
    divergences=True,
    var_names=[
        "chirp_mass",
        "mass_ratio",
        "chi_eff",
        "luminosity_distance",
        "delta_time",
        "phase",
        "psi",
        "cos_theta_jn",
        "ra",
        "dec",
    ][3:],
)
plt.savefig("corner.png")
plt.close()

import IPython
from utils import extract_extrinsic
# extract_extrinsic = jax.jit(extract_extrinsic)
#
IPython.embed()
# plot_corner(mcmc, labels, truths=start)
