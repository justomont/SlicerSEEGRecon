# SlicerAutoelectrodes
 
This is a 3D Slicer extension for semi-automatic placement of stereo electroencephalography (SEEG) from Annotation Markups.

## Installation

1. Download 3D Slicer from [https://download.slicer.org](https://download.slicer.org/).
2. Download the zip extension from https://github.com/justomont/SlicerAutoelectrodes.
3. Unzip the folder.
4. Start 3D Slicer.
5. Go to Modules (upper bar) and click on the magnifier icon.
6. Select the Extension Wizard > Extension Tools > Select Extension.
7. Select and Open the unzipped folder **SlicerAutoelectrodes**.
8. A window will appear
9. Make sure that both boxes are selected and then click on **Yes**.
10. Installation complete!
11. To open the extension, go to the Modules Menu > SEEG > Autoelectrodes.
12. Enjoy! 

## Tutorial

1. Start 3D Slicer.
2. Load a volume. 
3. Go to Markups > Create Markups > Point List.
4. Name the point list as **real-R** (if the electrodes are located in the right hemisphere) or **real-L** (if the electrodes are located in the left hemisphere).
5. For each one of the electrodes, place a point on the first (deepest) contact and in the last contact. **Important**: Both points should start with the same name, as that is the name that the electrode will have. The first point’s name must end in 1, while the last point’s name must end in the total number of contacts that electrode has. As an example, if we have an electrode with 10 contacts that is placed in the hippocampus, the first point could be named Hip1 and the last point Hip10.
    1. If an electrode is bent, more points must be added at each bending position specifing the number of the contact that is at the bending point. As an example, if our previous electrode was bent where the 6th contact is located, we must add 3 points: Hip1 (first contact), Hip6 (bending contact) and Hip10 (last contact).
6. Once that all the points corresponding to all the electrodes are placed, go to Modules Menu > SEEG > Autoelectrodes.
7. Click on Apply and the position of all the contacts will appear.
