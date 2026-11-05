# Neural population tuning statistics as priors for multitask generalization

This repo contains the source and figure generation code for Cohen & Drugowitsch, "Neural population tuning statistics as priors for multitask generalization". NeurIPS 2026. 

It is assumed that the user has standard scientific computing libraries (`numpy`, `scipy`, `matplotlib`, etc.) installed.

### File structure
---
- `src`: source files for all simulations
- `figs`: figure generation code (`fig_gen.py` in particular)

### High-level structure
---
All source code files are self-contained and can be run independently. Each one contains a main entry point routine that runs the respective experiment.

`figs/fig_gen.py` generates all panels for the main and supplementary text figures. As noted in that script, this code only generates panels for each figure. Final formatting in Illustrator was used to generate the figures that appear in the camera-ready version of the manuscript. The script permits generation of all figure panels at once, or for select generation of particular figures' panels. Please refer to the output of `./fig_gen.py -h` for instructions on how to generate specific figure
panels.
