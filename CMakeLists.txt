cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Autoelectrodes)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Autoelectrodes")
set(EXTENSION_CATEGORY "SEEG")
set(EXTENSION_CONTRIBUTORS "Justo Montoya (Universitat Pompeu Fabra)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Autoelectrodes.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Autoelectrodes/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Autoelectrodes)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
