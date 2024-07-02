# SlicerAutoelectrodes
 
Welcome to SlicerAutoelectrodes, a powerful 3D Slicer module designed to streamline the semi-automatic placement of stereo electroencephalography (SEEG) electrodes using Annotation Markups. Whether you're a researcher, clinician, or enthusiast in the neuroimaging field, this tool aims to enhance your workflow, saving you time and improving accuracy in electrode placement.

By leveraging the capabilities of 3D Slicer, SlicerAutoelectrodes allows you to effortlessly position SEEG electrodes with precision. Simply annotate your points, and let the extension handle the rest, providing you with reliable and reproducible electrode placements. Dive into our comprehensive tutorial to get started and unlock the full potential of your neuroimaging analysis.

## System Requirements

- Operating System: Windows 11, MacOS BigSur v11.6 (Linux is untested but expected to work)
- Software: 3D Slicer v5.0.3 or newer
- Dependencies: FreeSurfer Extension installed, with brain segmentation assigned the 'FreeSurferLabels' Lookup Table. To do this, select the segmentation volume in the "Volumes" module and navigate to "Display".

## Installation

1. Download 3D Slicer from their [website](https://download.slicer.org/).
2. Download the zip module from this repository.
3. Unzip the folder.
4. Start 3D Slicer.
5. Go to Modules (upper bar) and click on the magnifier icon.
6. Select Extension Wizard > Extension Tools > Select Extension.
7. Select and open the unzipped folder **SlicerAutoelectrodes**.
8. A window will appear
9. Ensure that both boxes are selected, then click on **Yes**.
10. Installation complete!
11. To open the extension, go to the Modules Menu > SEEG > Autoelectrodes.
12. Enjoy! 


## Tutorial

1. Start 3D Slicer.
2. Load a volume. 
3. Navigate to Markups > Create Markups > Point List.
4. Name the point list as real-R (for right hemisphere electrodes) or real-L (for left hemisphere electrodes).
5. For each electrode, place a point on the first (deepest) contact and the last contact. Important: Both points should start with the same name, indicating the electrode's name. The first point’s name must end in 1, and the last point’s name must end in the total number of contacts that electrode has. For example, for an electrode with 10 contacts placed in the hippocampus, you could name the first point _Hip1_ and the last point _Hip10_.
    1. If an electrode is bent, add more points at each bending position, specifying the contact number at the bend. For example, if the electrode is bent at the 6th contact, add points Hip1 (first contact), Hip6 (bending contact), and Hip10 (last contact).
6. Once all points for all electrodes are placed, go to Modules Menu > SEEG > Autoelectrodes.
7. Select the markup list that contains the contacts.
8. Click on Apply, and the position of all contacts will appear.

## Contributions and Support

We welcome any recommendations, bug reports, or requests. Please feel free to reach out to me directly on X (formerly Twitter) at [@Justo_Montoya_](https://x.com/Justo_Montoya_). Your feedback and contributions are invaluable in making this repository a robust resource for the neuroscience community.
