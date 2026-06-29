# SlicerSEEGRecon
[![BSD-3-Clause License](https://img.shields.io/github/license/justomont/SlicerSEEGRecon)](https://opensource.org/license/bsd-3-clause)
<img src="https://github.com/justomont/SlicerSEEGRecon/blob/main/SEEGRecon/Resources/Icons/SEEGRecon.png" align="right"
     alt="SlicerSEEGRecon Logo" width="160" height="178">
     
SEEG Reconstruction & Analysis Toolkit (a 3D Slicer Extension)

A practical, end-to-end tool to help you reconstruct and analyse SEEG implantations after surgery, directly inside [3D Slicer](https://www.slicer.org/).

This module is designed for clinicians and researchers working with intracranial EEG, reducing the manual steps needed to go from post-op scans → clean electrode localisation → publication-ready figures → exportable datasets.

<p align="center">
  <img src="https://github.com/justomont/SlicerSEEGRecon/blob/main/SEEGRecon/Resources/Icons/marca_vermella.png"
       alt="UPF logo"
       width="180">
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/justomont/SlicerSEEGRecon/blob/main/SEEGRecon/Resources/Icons/IMIM-logo.png"
       alt="IMIM logo"
       width="210">
</p>

## Table of Contents
- [What this tool helps you do](#-what-this-tool-helps-you-do)
- [Smart data import](#-smart-data-import-freesurfer-ready)
- [CT → MRI registration](#-ct--mri-registration-fully-automatic)
- [SEEG electrode reconstruction](#-semi-automatic-electrode-reconstruction)
- [MNI & anatomy mapping](#-mni--anatomy-mapping)
- [Visualization templates](#-figure-friendly-visualisation-templates)
- [Contact management](#-contact-management)
- [SEEG statistics](#-automatic-seeg-statistics)
- [Export projects](#-export-everything)
- [Requirements](#system-requirements)
- [Installation](#installation)
- [Open Source](#-open-source)
- [Authors](#authors)

## 🚀 What this tool helps you do

Instead of juggling multiple tools and manual steps, this extension lets you:

* 📥 Load MRI, CT, and FreeSurfer outputs automatically
* 🔗 Register CT to MRI in one click
* 🎯 Reconstruct SEEG electrodes semi-automatically
* 🧭 Map contacts to brain anatomy and MNI space
* 📊 Generate clinical reporting statistics instantly
* 🎨 Create publication-ready visualisation templates
* 📦 Export everything as a single portable project (.epiDB)

## Features
### 📥 Smart data import (FreeSurfer-ready)

Drop your patient folder in, and the module will try to figure things out for you.

It supports:

* FreeSurfer outputs (aseg.mgz, T1.mgz, surfaces, etc.)
* CT DICOM folders
* Standard MRI datasets
* Existing .epiDB projects

👉 Missing files? No problem. Everything available gets loaded automatically.

### 🔗 CT → MRI registration (fully automatic)

Align post-op CT with MRI in a single click.

* Built specifically for SEEG datasets
* Fast, robust, and automatic
* Saves a huge amount of manual work

⚠️ Still: always visually check the result (clinical sanity check recommended).

### 🎯 Semi-automatic electrode reconstruction

No more manual point-by-point reconstruction.

Just select:

* deepest contact (tip)
* superficial contact
* optional bend point
* electrode name

And the tool reconstructs the full trajectory for you.

It also:

* Computes monopolar contacts
* Generates bipolar derivations (A1–A2, etc.)
* Labels contacts using available brain segmentation (f.e. grey/white matter)

### 🧠 MNI & anatomy mapping

Automatically place your electrodes in standard space.

You get:

* MNI coordinates for all contacts
* Anatomical labels (atlas-based if available)
* Custom label map support
* A clean exportable table with everything organised

### 🎨 Figure-friendly visualisation templates

Because making figures shouldn’t take hours.

* Save your preferred visual style once
* Apply it to any patient in one click
* Includes a prebuilt template used in the IMIM-UPF Epilepsy Group

### 📍 Contact management

Small but very useful:

* Copy/paste contacts between lists
* Create subsets of clinically relevant electrodes
* Quickly reorganise your dataset without frustration

### 📊 Automatic SEEG statistics

Generate key descriptive metrics instantly:

* Total number of contacts
* Grey vs white matter distribution
* Left vs right hemisphere coverage
* Brain and hemisphere volumes (cm³)
* Contact density per region

👉 Ideal for methods sections and cohort description tables.

### 📦 Export everything

Save your entire reconstruction as a single project file.

The .epiDB format includes:

* Imaging data
* Electrode geometry
* Contact annotations
* Transformations
* Visualization settings
* Metadata

Think of it as a cleaner, more practical alternative to .mrb-style scenes.

## System Requirements

* Operating System: Windows 11, MacOS BigSur v11.6 to Tahoe v26.5.1 (Linux is untested but expected to work)
* [3D Slicer](https://www.slicer.org/) (recommended v5.8.1)

## Installation

1. Download 3D Slicer from their [website](https://download.slicer.org/).
2. Download the zip module from this repository.
3. Unzip the folder.
4. Start 3D Slicer.
5. Go to Modules (upper bar) and click on the magnifier icon.
6. Select Extension Wizard > Extension Tools > Select Extension.
7. Select and open the unzipped folder **SlicerSEEGRecon-main**.
8. A window will appear, ensure that both boxes are selected, then click on **Yes**.
9. Installation complete!
10. To open the extension, go to the Modules Menu > Electrophysiology > SEEGRecon.
11. Have fun! 

## 🤝 Open Source

This project is fully open source and built to grow. Contributions are more than welcome!

## Authors

- [@justomont](https://www.github.com/justomont): Justo Montoya-Gálvez (Department of Engineering, Universitat Pompeu Fabra (UPF), 08003 Barcelona, Spain)
- [@joricomico](https://www.github.com/joricomico): Alessandro Principe (Epilepsy Group, Hospital del Mar Research Institute (IMIM), 08003 Barcelona, Spain)

----
### Clinical Use Disclaimer

This software is intended to support research and clinical workflows involving stereoelectroencephalography (SEEG). Although some components have been validated on an internal dataset, the software is **not** a certified medical device.

Users are responsible for verifying all outputs before using them for clinical decision-making. The authors and contributors assume no responsibility for clinical decisions or patient outcomes resulting from the use of this software.

----

[Back to top](#top)
