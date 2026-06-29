cmake_minimum_required(VERSION 3.16.3)

project(SlicerSEEGRecon)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/justomont/SlicerSEEGRecon")
set(EXTENSION_CATEGORY "Electrophysiology")
set(EXTENSION_CONTRIBUTORS "Justo Montoya-Gálvez (Universitat Pompeu Fabra; UPF), Alessandro Principe (UPF)")
set(EXTENSION_DESCRIPTION "SEEG Reconstruction & Analysis Toolkit")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "SlicerFreeSurfer;SlicerElastix")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(SlicerSEEGRecon)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})