import sys
import time
from functools import partial

import gpucbc
gpucbc.activate_jax()  # noqa

import bilby
import matplotlib.pyplot as plt
import numpy as np
import ptemcee

import arviz
from gpucbc.waveforms import TF2
from jax import jit
from jax import numpy as jnp
from jax.random import PRNGKey
import numpyro
from numpyro.infer import NUTS, MCMC
from numpyro import distributions, handlers
from numpyro.distributions import ProjectedNormal
from numpyro.infer.reparam import ProjectedNormalReparam

import utils
from utils import (
    compute_evidences,
    generate_burst,
    generate_waveform,
    extrinsic_ln_prior,
    extrinsic_priors,
    ln_likelihood,
    plot_corner,
    run_sampler,
    set_data,
)
from model import model, burst_intrinsic_prior

duration = int(sys.argv[1])

priors = bilby.core.prior.PriorDict()
priors["log_amplitude"] = bilby.core.prior.Uniform(-22, -20)
priors["width"] = bilby.core.prior.Uniform(2, 20)
priors["centre_frequency"] = bilby.core.prior.Uniform(60, 80)
priors.update(extrinsic_priors())
ndim = len(priors)
# true_parameters = jnp.asarray(list(priors.sample().values()))
true_parameters = jnp.asarray([
    -21.0,  15.0,  70.0, -0.01, 0.2, 0.4, 0.3, 0.9, 0.5,
])
# priors["log_amplitude"].minimum = -25
# priors["log_amplitude"].maximum = -17
priors["width"].maximum = 100
search_keys = list(priors.keys())
labels = [key.replace("_", " ") for key in priors]
minima = jnp.asarray([priors[key].minimum for key in search_keys])
maxima = jnp.asarray([priors[key].maximum for key in search_keys])
domain = maxima - minima
fmin = int(true_parameters[2]) - 10
fmax = int(true_parameters[2]) + 10
utils.REFERENCE_FREQUENCY = true_parameters[2]
model.REFERENCE_FREQUENCY = true_parameters[2]

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1", "K1"][:3])
for ifo in ifos:
    ifo.minimum_frequency = fmin
    ifo.maximum_frequency = fmax
# ifos.set_strain_data_from_zero_noise(
ifos.set_strain_data_from_power_spectral_densities(
    duration=duration, sampling_frequency=2048
)
frequencies = jnp.arange(fmin, fmax + 1 / duration, 1 / duration)
TF2.update_cache(frequency_array=frequencies)
generator = partial(generate_waveform, wfg=generate_burst, frequencies=frequencies)
data, asd = set_data(ifos, generator, true_parameters)
start = true_parameters
kwargs = dict(
    wfg=generator,
    data=data,
    asd=asd,
    duration=duration,
    intrinsic_prior=burst_intrinsic_prior,
    # init_params=start,
)

# @handlers.reparam(config={
#     "orientation": ProjectedNormalReparam(),
#     "sky": ProjectedNormalReparam(),
# })
# def model():
#     delta_time = numpyro.sample("delta_time", distributions.Uniform(-0.5, 0.5))
#     # sky_x = numpyro.sample("sky_x", distributions.Normal())
#     # sky_y = numpyro.sample("sky_y", distributions.Normal())
#     # sky_z = numpyro.sample("sky_z", distributions.Normal())
#     # orientation_w = numpyro.sample("orientation_w", distributions.Normal())
#     # orientation_x = numpyro.sample("orientation_x", distributions.Normal())
#     # orientation_y = numpyro.sample("orientation_y", distributions.Normal())
#     # orientation_z = numpyro.sample("orientation_z", distributions.Normal())
#     sky = numpyro.sample("sky", ProjectedNormal(jnp.zeros(3)))
#     orientation = numpyro.sample("orientation", ProjectedNormal(jnp.zeros(4)))
#     numpyro.factor("log_likelihood", -delta_time**2)
#     numpyro.deterministic("ln_likelihood", -delta_time**2)
#     # return delta_time, sky_x, sky_y, sky_z, orientation_w, orientation_x, orientation_y, orientation_z
#     # return delta_time, *sky, *orientation
# kwargs = dict()
print(ln_likelihood(true_parameters, generator, data, asd, duration))


kernel = NUTS(
    model,
    adapt_mass_matrix=True,
    dense_mass=True,
)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000)
mcmc.run(PRNGKey(np.random.randint(0, 1000000)), beta=1, **kwargs)
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
    var_names=["log_amplitude", "width", "centre_frequency", "delta_time", "phase", "psi", "cos_theta_jn", "ra", "dec"],
)
plt.savefig("corner.png")
plt.close()
import corner
for key in ["sky", "orientation"]:
    corner.corner(np.array(mcmc.get_samples()[f"{key}_normal"]))
    plt.savefig(f"{key}_corner.png")
    plt.close()
import IPython; IPython.embed()
import sys; sys.exit()


def run_mcmc(beta):
    mcmc.run(
        PRNGKey(np.random.randint(0, 100000)),
        beta=beta,
        **kwargs,
    )
    ln_ls = np.array(mcmc.get_samples()["ln_likelihood"])
    return ln_ls, mcmc


def plot_evidences(beta, step, thermo):
    fig, axes = plt.subplots(
        nrows=3,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1, 1]),
    )
    plt.sca(axes[0])
    plt.plot(beta, np.insert(step, 0, 0), label="Stepping stone")
    plt.plot(beta, np.insert(thermo, 0, 0), label="Thermodynamic")
    plt.legend(loc="upper left")
    plt.ylabel("$\\ln\\mathcal{Z}_{\\beta}$")
    plt.sca(axes[1])
    plt.plot(beta, np.insert(step, 0, 0), label="Stepping stone")
    plt.plot(beta, np.insert(thermo, 0, 0), label="Thermodynamic")
    plt.ylim(min(1.1 * min(np.min(step), np.min(thermo)), 0), 1.1 * abs(min(np.min(step), np.min(thermo))))
    plt.ylabel("$\\ln\\mathcal{Z}_{\\beta}$")
    plt.sca(axes[2])
    plt.plot(beta, np.insert(step - thermo, 0, 0))
    for value in beta:
        plt.axvline(value, color="C0")
    plt.xlabel("$\\beta$")
    plt.ylabel("$\\Delta\\ln\\mathcal{Z}_{\\beta}$")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("evidence.png")
    plt.close()


chains = list()
all_lnls = list()
betas = ptemcee.sampler.default_beta_ladder(ndim, 2)[::-1]
tstart = time.time()
for beta in betas:
    ln_ls, mcmc = run_mcmc(beta)
    mcmc.print_summary(exclude_deterministic=False)
    all_lnls.append(ln_ls)
    chains.append(mcmc.get_samples())
tmid = time.time()
# labels = [
#     "log_amplitude",
#     "width",
#     "centre_frequency",
#     "sky[0]",
#     "sky[1]",
#     "sky[2]",
#     "orientation[0]",
#     "orientation[1]",
#     "orientation[2]",
#     "orientation[3]",
# ]
# plot_corner(mcmc, labels, start)
# utils.extrinsic_corner(mcmc, start)
all_lnls = np.array(all_lnls)
step_cumulative, thermo_cumulative = compute_evidences(betas, all_lnls)
differences = step_cumulative - thermo_cumulative
plot_evidences(betas, step_cumulative, thermo_cumulative)
import IPython; IPython.embed()

evidence_threshold = 0.05

while max(abs(np.diff(differences, prepend=0))) > evidence_threshold:
    min_idx = np.argmax(abs(np.diff(differences, prepend=0)))
    if min_idx == 0:
        beta = betas[0] * 0.5
    else:
        min_idx += 1
        beta = (betas[min_idx - 1] + betas[min_idx]) * 0.5
    print(
        f"{beta:.3f}",
        min_idx,
        f"{max(abs(np.diff(differences, prepend=0))):.3f}",
        sum(abs(np.diff(differences, prepend=0)) > evidence_threshold),
    )
    ln_ls, mcmc = run_mcmc(beta)
    betas = np.insert(betas, min_idx, beta)
    all_lnls = np.insert(all_lnls, min_idx, ln_ls, 0)
    chains.insert(min_idx, mcmc)
    step_cumulative, thermo_cumulative = compute_evidences(betas, all_lnls)
    differences = step_cumulative - thermo_cumulative
    plot_evidences(betas, step_cumulative, thermo_cumulative)
tend = time.time()
print(f"Initial sampling time: {tmid - tstart:.2f}s")
print(f"Total sampling time: {tend - tstart:.2f}s")
import IPython

IPython.embed()
