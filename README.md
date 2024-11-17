[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<p align="center">
  <img src="https://github.com/AlfredoCamaraC/MDyn---Accelerated-Modal-Dynamics-Solver/blob/main/mdyn_logo.png" alt="" width="300">
</p>

# MDyn: Accelerated Modal Dynamics Solver
MDyn is a fast modal superposition analysis software highly vectorised in Python to solve structural dynamics problems involving wind, pedestrians, vehicles, trains, ... and their interaction.

## Getting Started
The running codes are fully commented and should be easy to navigate, but tutorials are on the way.

## Instalation
No instalation is needed (beyond some standard Python libraries like numpy, scipy, matplotlib). The subfolder "code" needs to be in the working directory when running the main python script.

## Accompanying Papers
Please refer to the paper (https://doi.org/10.1016/j.advengsoft.2020.102934) for details of the MDyn solver, validation, and more information about the library.

Details about the use of MDyn to solve pedestrian-induced vibrations in footbridges are included in this paper: https://doi.org/10.1016/j.istruc.2024.106263.

Information on the use of MDyn coupled with image processing of CCTV in bridges to obtain the response of bridges under real traffic is included in this paper: https://doi.org/10.1016/j.engstruct.2024.118653.

The use of MDyn with microsimulated traffic flows is presented in this paper: https://doi.org/10.1142/S0219455425502608.

The use of MDyn to simulate the wind-vehicle-bridge interaction under skew winds is described in this paper: https://doi.org/10.1016/j.jweia.2021.104672.

## Citing MDyn

If you use MDyn in your research, please cite the following paper:

```
@article{camara2021,
  title={A fast mode superposition algorithm and its application to the analysis of bridges under moving loads},
  author={Camara A},
  journal={Advances in Engineering Software},
  volume={151},
  pages={102934},
  year={2021},
  publisher={Elsevier}
}
```
## Showcase

<p align="center">
  <img src="https://github.com/AlfredoCamaraC/MDyn---Accelerated-Modal-Dynamics-solver/blob/main/deformation_abaqus_MDyn.png" alt="" width="600">
</p>
<p align="center">
  Comparison between the deformation induced by a vehicle crossing the Queensferry Bridge (Scotland) using ABAQUS and MDyn.
</p>

<br>


<p align="center">
  <img src="https://github.com/AlfredoCamaraC/MDyn---Accelerated-Modal-Dynamics-Solver/blob/main/MDyn_LQS_resultsTMD.gif" width="800">
</p>
<p align="center">
  Linear quasi-steady analysis of the wind-induced response in a footbridge with a TMD at the centre.
</p>

<br>

<p align="center">
  <img src="https://github.com/AlfredoCamaraC/MDyn---Accelerated-Modal-Dynamics-Solver/blob/main/MDyn_resonant_pedestrian_footbridge.gif" width="800">
</p>
<p align="center">
  Vibration of a footbridge under the action of a resonant pedestrian.
</p>

<p align="center">
  <img src="https://github.com/AlfredoCamaraC/MDyn---Accelerated-Modal-Dynamics-solver/blob/main/displacements_footbridge_resonance.png" alt="" width="600">
</p>
<p align="center">
Vertical displacement at midspan</p>
