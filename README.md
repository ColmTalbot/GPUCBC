[![Build Status](https://travis-ci.com/ColmTalbot/GPUCBC.svg?branch=master)](https://travis-ci.com/ColmTalbot/GPUCBC) [![Maintainability](https://api.codeclimate.com/v1/badges/be3b54493b6b3cd4c48a/maintainability)](https://codeclimate.com/github/ColmTalbot/GPUCBC/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/be3b54493b6b3cd4c48a/test_coverage)](https://codeclimate.com/github/ColmTalbot/GPUCBC/test_coverage)

# GPUCBC

Code for CBC parameter estimation with GPUs

This currently contains a `bilby` compatible likelihood and `waveform_generator` along with an implementation of the post-Newtonian inspiral waveform `TaylorF2` with aligned spins and tidal effects.

# Attribution

Please cite [Talbot _et al_ (2019)](https://doi.org/10.1103/PhysRevD.100.043030) if you find this package useful.

```
@ARTICLE{2019PhRvD.100d3030T,
       author = {{Talbot}, Colm and {Smith}, Rory and {Thrane}, Eric and
         {Poole}, Gregory B.},
        title = "{Parallelized inference for gravitational-wave astronomy}",
      journal = {\prd},
         year = 2019,
        month = aug,
       volume = {100},
       number = {4},
          eid = {043030},
        pages = {043030},
          doi = {10.1103/PhysRevD.100.043030},
archivePrefix = {arXiv},
       eprint = {1904.02863},
 primaryClass = {astro-ph.IM},
}
```
