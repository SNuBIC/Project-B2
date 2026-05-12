## Reproducibility Materials 
for **G. Grünwald, L. Hensel, M. Deisenhofer, S. Lautenbach, K. Kormann, R. Grauer, Solving the six-dimensional Vlasov-Maxwell System with Active Flux and Splitting Methods, arXiv preprint, 2025, [arXiv:2511.22440](https://arxiv.org/abs/2511.22440)**

The numerical results in this publication are based on:

The muphyII Code: Multiphysics Plasma Simulation on Large HPC Systems <br>
F. Allmann-Rahn, S. Lautenbach, M. Deisenhofer, R. Grauer <br>
CPC 296 (2024) 109064 https://doi.org/10.1016/j.cpc.2023.109064

The muphyII code is available on <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub (https://github.com/muphy2-framework/muphy2) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Zenodo (https://zenodo.org/doi/10.5281/zenodo.8061586). <br>

Information on how to compile, configure and run the code can be found in 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; muphy2/doc/muphy2_documentation.pdf

All modifications are documented in 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ActiveFlux_VM.patch

which can be applied to the upstream source via `git apply ActiveFlux_VM.patch`.
