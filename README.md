# Geo_DECT

Geo_DECT is a plugin for Dragonfly 3D that converts dual-energy CT (DECT) data into 3D maps of effective atomic number (Zeff) and electron density (ρe), using the coefficients of a stoichiometric calibration. It was developed for geological samples scanned with medical CT or micro-CT.

<img width="1097" height="590" alt="overview" src="https://github.com/user-attachments/assets/8cf76532-d63a-49cd-81b8-ba849249bf16" />


## Main features

- Calculation of Zeff and ρe from a LOW-energy and a HIGH-energy dataset (TIFF or DICOM)
- Three 3D volumes (Zeff, ρe and combined) that behave as native Dragonfly objects
- Interactive colouring of any Zeff or ρe range, in 2D and 3D
- Export of statistics and crossplots (PDF), probed points (CSV) and test-slice maps (TIFF)

## Requirements

- Windows
- Dragonfly 3D 2025.1 (or compatible), installed separately with its own license
- matplotlib in the Dragonfly Python environment (needed only for the PDF exports)
- Tested on: Intel Xeon Silver 4216, 128 GB RAM, NVIDIA RTX A4000 (16 GB)
- Datasets below about 2 billion voxels are recommended; larger datasets can be cropped or binned 2x2x2 (Average) in ImageJ/Fiji

## Installation

1. Close Dragonfly 3D.
2. Download this repository and extract it.
3. Copy the entire folder `Geo_DECT_7e90f2c77eb011f18afab07b2518f399` into `%localappdata%\Comet\Dragonfly2025.1\pythonUserExtensions\Plugins\`
4. Optional, for the PDF exports, run in a command prompt: `"C:\Program Files\Dragonfly\Python_env\python.exe" -m pip install matplotlib`
5. Launch Dragonfly 3D and open the plugin from Utilities > Open Plugins > Geo_DECT.

Do not rename the folder and do not use the Dragonfly Python Plugin Generator: copy the folder as it is.

## User Guide

The complete step-by-step workflow, with all inputs, outputs, options and troubleshooting, is described in the [User Guide](User_Guide_Geo_DECT.pdf).

## Example data

The dual-energy CT datasets used in the Geo_DECT article (an oolitic hematite ironstone scanned by micro-CT and a varved lacustrine sediment core scanned by medical CT), together with the calibration coefficients and the Geo_DECT parameters, are available on Zenodo: https://doi.org/10.5281/zenodo.24105517

## Background references

- Martini, M., Francus, P., Di Schiavi Trotta, L., Després, P., 2021. Identification of common minerals using stoichiometric calibration method for dual-energy CT. Geochemistry, Geophysics, Geosystems 22(11), e2021GC009885. https://doi.org/10.1029/2021GC009885
- Martini, M., Francus, P., Schröer, L., Buyse, F., Kibleur, P., Cnudde, V., Di Schiavi Trotta, L., Després, P., 2025. Characterisation of geological samples with dual-energy XCT: a comparison of three different scanners. Geostandards and Geoanalytical Research 49(3), 645-664. https://doi.org/10.1111/ggr.12608

## License

MIT License (see the LICENSE file). Dragonfly 3D is not part of this repository and requires its own license.

## Contact

Margherita Martini, Laboratoire de Géologie de Lyon - Terre, Planètes, Environnement (LGL-TPE).
Questions and bug reports: please use the Issues page of this repository.

