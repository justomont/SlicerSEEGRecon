import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import re
import numpy as np
import csv
from itertools import compress

import os
from os import listdir
from os.path import isfile, join
try:
    import pandas as pd
except: 
    import pip
    pip.main(["install","pandas"])
    import pandas as pd

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def anatomicREL(tag):
    region = ['Unknown',
               'Left-Cerebral-White-Matter',
               'Left-Cerebral-Cortex',
               'Left-Lateral-Ventricle',
               'Left-Inf-Lat-Vent',
               'Left-Cerebellum-White-Matter',
               'Left-Cerebellum-Cortex',
               'Left-Thalamus-Proper',
               'Left-Caudate',
               'Left-Putamen',
               'Left-Pallidum',
               '3rd-Ventricle',
               '4th-Ventricle',
               'Brain-Stem',
               'Left-Hippocampus',
               'Left-Amygdala',
               'CSF',
               'Left-Accumbens-area',
               'Left-VentralDC',
               'Left-vessel',
               'Left-choroid-plexus',
               'Right-Cerebral-White-Matter',
               'Right-Cerebral-Cortex',
               'Right-Lateral-Ventricle',
               'Right-Inf-Lat-Vent',
               'Right-Cerebellum-White-Matter',
               'Right-Cerebellum-Cortex',
               'Right-Thalamus-Proper',
               'Right-Caudate',
               'Right-Putamen',
               'Right-Pallidum',
               'Right-Hippocampus',
               'Right-Amygdala',
               'Right-Accumbens-area',
               'Right-VentralDC',
               'Right-vessel',
               'Right-choroid-plexus',
               '5th-Ventricle',
               'WM-hypointensities',
               'non-WM-hypointensities',
               'Optic-Chiasm',
               'CC Posterior',
               'CC Mid Posterior',
               'CC Central',
               'CC Mid Anterior',
               'CC Anterior']
    
    indx = [0,
            2,
            3,
            4,
            5,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            24,
            26,
            28,
            30,
            31,
            41,
            42,
            43,
            44,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            54,
            58,
            60,
            62,
            63,
            72,
            77,
            80,
            85,
            251,
            252,
            253,
            254,
            255] 
    
    anatomic = region[indx.index(tag)]
    return anatomic

def RAStoIJK(ras,volumeNode):
    transformRasToVolumeRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
    point_VolumeRas = transformRasToVolumeRas.TransformPoint(ras[0:3])
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
    point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
    return point_Ijk

def NthFiducialPosition(fidNode,n):
    pos = [0,0,0]
    fidNode.GetNthFiducialPosition(n,pos)
    return pos
    

def findContacts():
    
    hemisphere_location = ["L","R"]
    Hpresent = []
    
    for Hloc in hemisphere_location:
        fidNode = slicer.mrmlScene.GetFirstNodeByName("real-"+Hloc)
        if fidNode != None:
            Hpresent.append(Hloc)

    for Hloc in Hpresent:
        # Get the markup fiducials
        fidNode = slicer.mrmlScene.GetFirstNodeByName("real-"+Hloc)
        
        # Get the aseg map
        volumeNode = slicer.mrmlScene.GetFirstNodeByName('aseg')
        voxelArray = slicer.util.arrayFromVolume(volumeNode)
        
        # All markup's names and positions in RAS coordinates
        markup_names = [fidNode.GetNthFiducialLabel(i) for i in range(fidNode.GetNumberOfFiducials())]
        markup_RAS = [NthFiducialPosition(fidNode,i) for i in range(fidNode.GetNumberOfFiducials())]
        
        # Remove markups that signal the END of the electrode (wich is not the last contact but the tip of the elctrode, outside of the skull)
        boolean = [has_numbers(name) for name in markup_names]
        clean_markup_names = list(compress(markup_names, boolean))
        clean_markup_RAS = list(compress(markup_RAS, boolean))
        
        # Sort the lists alphabetically
        tuples = zip(*sorted(zip(clean_markup_names, clean_markup_RAS)))
        markups, RAS = [list(tuple) for tuple in  tuples]
        
        # Store also the location of the end of the electrodes just for representational purposes
        end_boolean = [not element for element in boolean]
        end_markup_names = list(compress(markup_names, end_boolean))
        end_markup_RAS = list(compress(markup_RAS, end_boolean))
        end_tuples = zip(*sorted(zip(end_markup_names, end_markup_RAS)))
        end_markups, end_RAS = [list(tuple) for tuple in  end_tuples]
        
        #############################################################################
        # Monopolar. The tool just adds the contacts where they really are in space.#
        #############################################################################
        # Goal: Compute the position of the remaining contacts and include them in the monopolar markup list (real-R/L)
        
        # Initialize lists for the markups and their RAS coordinates
        monopolar_markups = []
        monopolar_RAS = []
        
        monopolar_markups_WM = []
        monopolar_RAS_WM = []
        fidNodeWM = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeWM.SetName("real-"+Hloc+"-WM")
        slicer.mrmlScene.AddNode(fidNodeWM)
        
        monopolar_markups_P = []
        monopolar_RAS_P = []
        fidNodeP = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeP.SetName("real-"+Hloc+"-P")
        slicer.mrmlScene.AddNode(fidNodeP)
        
        monopolar_markups_E = []
        monopolar_RAS_E = []
        fidNodeE = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeE.SetName("real-"+Hloc+"-ends")
        slicer.mrmlScene.AddNode(fidNodeE)
        
        # Iterate over all the markups defined by the user
        for index,markup in enumerate(markups): 
            if index < len(markups)-1: # There's a -1 one here because the penultimate markup adds the last markup, so there is no need to check the last one
            
                # Check the letter/name and the number of the selected markup and the next one in the list
                letter = ''.join([i for i in markup if not i.isdigit()])
                digit = int(re.search(r'\d+', markup).group())
                next_letter = ''.join([i for i in markups[index+1] if not i.isdigit()])
                next_digit = int(re.search(r'\d+', markups[index+1]).group())
                
                # If they have the same letter we can continue because it means that are part of the same electrode
                if letter == next_letter:
                    # add initial contact to the new list
                    if markup not in monopolar_markups:
                        monopolar_markups.append(markup)
                        monopolar_RAS.append(RAS[index])
                        # check location of the contact (WM or not)
                        ijk_position = RAStoIJK(RAS[index], volumeNode)
                        anatomic_position = anatomicREL(voxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                        if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                            monopolar_markups_WM.append(markup)
                            monopolar_RAS_WM.append(RAS[index])
                            fidNodeWM.AddFiducialFromArray(RAS[index], markup)
                        else:
                            monopolar_markups_P.append(markup)
                            monopolar_RAS_P.append(RAS[index])
                            fidNodeP.AddFiducialFromArray(RAS[index], markup)
                    # calculate how many markups should be added until the next user-defined markup
                    additions = next_digit - digit-1
                    # calculate how distant the markups should be
                    distance = np.subtract(RAS[index+1], RAS[index])/(additions+1)
                    # add these new markups
                    for i in range(additions):
                        new_digit = digit+i+1
                        new_position = np.add(RAS[index], distance*(digit+i))
                        monopolar_markups.append(letter+str(new_digit))
                        monopolar_RAS.append(new_position)
                        
                        fidNode.AddFiducialFromArray(new_position, letter+str(new_digit))
                        
                        ijk_position = RAStoIJK(new_position, volumeNode)
                        anatomic_position = anatomicREL(voxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                        if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                            monopolar_markups_WM.append(letter+str(new_digit))
                            monopolar_RAS_WM.append(new_position)
                            fidNodeWM.AddFiducialFromArray(new_position, letter+str(new_digit))
                        else:
                            monopolar_markups_P.append(letter+str(new_digit))
                            monopolar_RAS_P.append(new_position)
                            fidNodeP.AddFiducialFromArray(new_position, letter+str(new_digit))
                        
                    monopolar_markups.append(markups[index+1])
                    monopolar_RAS.append(RAS[index+1])
                    ijk_position = RAStoIJK(new_position, volumeNode)
                    anatomic_position = anatomicREL(voxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                    if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                        monopolar_markups_WM.append(markups[index+1])
                        monopolar_RAS_WM.append(RAS[index+1])
                        fidNodeWM.AddFiducialFromArray(RAS[index+1], markups[index+1])
                    else:
                        monopolar_markups_P.append(markups[index+1])
                        monopolar_RAS_P.append(RAS[index+1])
                        fidNodeP.AddFiducialFromArray(RAS[index+1], markups[index+1])
                    
                    #Create rulers where the whole electrodes are
                    rulerNode = slicer.vtkMRMLAnnotationRulerNode()
                    rulerNode.SetName(letter)
                    rulerNode.Initialize(slicer.mrmlScene)
                    rulerNode.SetPosition1(RAS[index])
                    rulerNode.SetPosition2(RAS[index+1])
                    if Hloc == "R":
                        rulerNode.GetDisplayNode().SetColor([0,0,1.])
                    else:
                        rulerNode.GetDisplayNode().SetColor([170/255,0,0])
                    rulerNode.SetDistanceAnnotationScale(0)
                    rulerNode.GetDisplayNode().SetLineThickness(6)
                    rulerNode.GetDisplayNode().SetMaxTicks(0)
                    rulerNode.SetLocked(True)
                
                # if the letter is not the same as the next one it means we found the last markup of the electrode, 
                # thus we can extend this last section to better grpahically represent the electrode
                else:
                    # compute the vector that defines the line that passes through the last point and the penultimate user-defined markup
                    pointA = RAS[index-1]
                    pointB = RAS[index]
                    l = pointB[0]-pointA[0]
                    m = pointB[1]-pointA[1]
                    n = pointB[2]-pointA[2]
                    AB = np.array([l,m,n])
                    # select the last point
                    pointE = end_RAS[end_markups.index(letter)]
                    l = pointE[0]-pointA[0]
                    m = pointE[1]-pointA[1]
                    n = pointE[2]-pointA[2]
                    AE = np.array([l,m,n])
                    # project the last markup (E) onto the line generated by AB
                    P = pointA + np.dot(AE,AB) / np.dot(AB,AB) * AB
                    # generate projection on Slicer 
                    fidNodeE.AddFiducialFromArray(P,letter)
                    # add ruler
                    rulerNode = slicer.vtkMRMLAnnotationRulerNode()
                    rulerNode.SetName(letter)
                    rulerNode.Initialize(slicer.mrmlScene)
                    rulerNode.SetPosition1(RAS[index])
                    rulerNode.SetPosition2(P)
                    if Hloc == "R":
                        rulerNode.GetDisplayNode().SetColor([0,0,1.])
                    else:
                        rulerNode.GetDisplayNode().SetColor([170/255,0,0])
                    rulerNode.SetDistanceAnnotationScale(0)
                    rulerNode.GetDisplayNode().SetLineThickness(6)
                    rulerNode.GetDisplayNode().SetMaxTicks(0)
                    rulerNode.SetLocked(True)
                    
        # Lock all markups
        for markupN in range(fidNode.GetNumberOfMarkups()):
            fidNode.SetNthFiducialLocked(markupN,True)
        for markupN in range(fidNodeWM.GetNumberOfMarkups()):
            fidNodeWM.SetNthFiducialLocked(markupN,True)
        for markupN in range(fidNodeP.GetNumberOfMarkups()):
            fidNodeP.SetNthFiducialLocked(markupN,True)
        for markupN in range(fidNodeE.GetNumberOfMarkups()):
            fidNodeE.SetNthFiducialLocked(markupN,True)
        
        logging.info("Monopolar contact placement complete.\n") 
        
        # Atlases
        # ASEG atlas
        for index,contact in enumerate(monopolar_markups):
            ras = monopolar_RAS[index]
            point_ijk = RAStoIJK(ras,volumeNode)
            aseg_label = anatomicREL(voxelArray[point_ijk[2],point_ijk[1],point_ijk[0]])
            # print(contact+" "+aseg_label+"\n")
       
        # # ICBM152  
        # # Set parameters
        # mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
        # templatePath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
        # labelmapPath = os.path.join(mniPath, 'mni_icbm152_CerebrA_tal_nlin_sym_09c.nii')
        # movingmaskPath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')
        
        # fixedVolumeNode = slicer.mrmlScene.GetFirstNodeByName("brain")
        # movingVolumeNode = slicer.util.loadVolume(templatePath,properties={"name":"ICBM152_T1","center":True})
        # labelmapVolumeNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_labels","labelmap":True,"center":True})
        
        # linearTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        # linearTransformNode.SetName("Transform2MNI")
        # outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # outputVolumeNode.SetName("ICBM152_registered")
        
        # fixedmaskNode = slicer.mrmlScene.GetFirstNodeByName("aseg")
        # movingmaskNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_mask","labelmap":True,"center":True})
        
        # parameters = {}
        # parameters["fixedVolume"] = fixedVolumeNode
        # parameters["movingVolume"] = movingVolumeNode
        # # parameters["samplingPercentage"] = 0.002  # Default sampling percentage of 0.002, to change it, uncomment the next line and change the value
        # # parameters["splineGridSize"] = [14,10,12] # Default B-spline grid size 14,10,12, to change it, uncomment the next line and change the value
        # parameters["linearTransform"] = linearTransformNode
        # parameters["outputVolume"] = outputVolumeNode
        # parameters["initializeTransformMode"] = "useCenterOfHeadAlign"
        # parameters["useRigid"] = True
        # parameters["useScaleVersor3D"] = True
        # parameters["useScaleSkewVersor3D"] = True
        # parameters["useAffine"] = True
        # parameters["maskProcessingMode"] = "ROI"
        # parameters["fixedBinaryVolume"] = fixedmaskNode
        # parameters["movingBinaryVolume"] = movingmaskNode
        # parameters["outputFixedVolumeROI"] = fixedmaskNode
        # parameters["outputMovingVolumeROI"] = fixedmaskNode
        # # Execution
        # generalRegistration = slicer.modules.brainsfit
        # cliNode = slicer.cli.run(generalRegistration, None, parameters)
        
        # logging.info("enter.\n")
        # transform = slicer.util.getFirstNodeByName("Transform2MNI")
        # labels = slicer.util.getFirstNodeByName("MNI_labels")
        # labels.ApplyTransformMatrix(transform.GetMatrixTransformToParent())
        # transform = slicer.util.getFirstNodeByName("Transform2MNI")
        # labels = slicer.util.getFirstNodeByName("MNI_labels")
        # labels.ApplyTransformMatrix(transform.GetMatrixTransformToParent())
        # logging.info("out.\n")
       
        # Bipolar 
        fidNodeBi = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi.SetName("Bi-real-"+Hloc)
        slicer.mrmlScene.AddNode(fidNodeBi)
        bipolar_markups = []
        bipolar_RAS = []
        
        fidNodeBi_WM = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi_WM.SetName("Bi-real-"+Hloc+"_WM")
        slicer.mrmlScene.AddNode(fidNodeBi_WM)
        bipolar_markups_WM = []
        bipolar_RAS_WM = []
        
        fidNodeBi_P = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi_P.SetName("Bi-real-"+Hloc+"_P")
        slicer.mrmlScene.AddNode(fidNodeBi_P)
        bipolar_markups_P = []
        bipolar_RAS_P = []
        
        
        for index,markup in enumerate(monopolar_markups):
            if index < len(monopolar_markups)-1:
                letter = ''.join([i for i in markup if not i.isdigit()])
                digit = int(re.search(r'\d+', markup).group())
                next_letter = ''.join([i for i in monopolar_markups[index+1] if not i.isdigit()])
                next_digit = int(re.search(r'\d+', monopolar_markups[index+1]).group())
                
                if letter == next_letter:
                    middle_point = np.add(monopolar_RAS[index+1], monopolar_RAS[index])/2
                    bi_tag = "-".join([markup,monopolar_markups[index+1]])
                    bipolar_markups.append(bi_tag)
                    bipolar_RAS.append(middle_point)
                    
                    fidNodeBi.AddFiducialFromArray(middle_point, bi_tag)
                    
                    ijk_position = RAStoIJK(middle_point, volumeNode)
                    anatomic_position = anatomicREL(voxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                    if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                        bipolar_markups_WM.append(bi_tag)
                        bipolar_RAS_WM.append(middle_point)
                        fidNodeBi_WM.AddFiducialFromArray(middle_point, bi_tag)
                    else:
                        bipolar_markups_P.append(bi_tag)
                        bipolar_RAS_P.append(middle_point)
                        fidNodeBi_P.AddFiducialFromArray(middle_point, bi_tag)
                    
        # Lock all markups
        for markupN in range(fidNodeBi.GetNumberOfMarkups()):
            fidNodeBi.SetNthFiducialLocked(markupN,True)
        for markupN in range(fidNodeBi_WM.GetNumberOfMarkups()):
            fidNodeBi_WM.SetNthFiducialLocked(markupN,True)
        for markupN in range(fidNodeBi_P.GetNumberOfMarkups()):
            fidNodeBi_P.SetNthFiducialLocked(markupN,True)
        
        logging.info("Bipolar contact placement complete.\n") 
        
def regionsMNI():
    
    # ICBM152  
    # Paths
    mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
    templatePath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_sym_09a.nii') # moving volume
    movingmaskPath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii') # moving volume mask
    labelmapPath = os.path.join(mniPath, 'mni_icbm152_CerebrA_tal_nlin_sym_09c.nii') # labelmap
    
    # Set parameters
    fixedVolumeNode = slicer.mrmlScene.GetFirstNodeByName("brain")
    movingVolumeNode = slicer.util.loadVolume(templatePath,properties={"name":"ICBM152_T1","center":True})
    
    linearTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    linearTransformNode.SetName("Transform2MNI")
    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolumeNode.SetName("ICBM152_registered")
    
    fixedmaskNode = slicer.mrmlScene.CopyNode(slicer.util.getFirstNodeByName("aseg"))
    fixedmaskNode.SetName("aseg_mask")
    slicer.mrmlScene.AddNode(fixedmaskNode)
    movingmaskNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_mask","labelmap":True,"center":True})
    
    parameters = {}
    parameters["fixedVolume"] = fixedVolumeNode
    parameters["movingVolume"] = movingVolumeNode
    parameters["samplingPercentage"] = 0.005  
    parameters["linearTransform"] = linearTransformNode
    parameters["outputVolume"] = outputVolumeNode
    parameters["initializeTransformMode"] = "useCenterOfHeadAlign"
    parameters["useRigid"] = True
    parameters["useScaleVersor3D"] = True
    parameters["useScaleSkewVersor3D"] = True
    parameters["useAffine"] = True
    parameters["maskProcessingMode"] = "ROI"
    parameters["fixedBinaryVolume"] = fixedmaskNode
    parameters["movingBinaryVolume"] = movingmaskNode
    parameters["outputFixedVolumeROI"] = fixedmaskNode
    parameters["outputMovingVolumeROI"] = fixedmaskNode
    
    # Execution
    generalRegistration = slicer.modules.brainsfit
    cliNode = slicer.cli.run(generalRegistration, None, parameters)
    
    labelmapVolumeNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_labels","labelmap":True,"center":True})
    transform = slicer.util.getFirstNodeByName("Transform2MNI")
    transformedLabels = slicer.mrmlScene.CopyNode(labelmapVolumeNode)
    transformedLabels.SetName("transformed_MNI_labels")
    transformedLabels.ApplyTransformMatrix(transform.GetMatrixTransformToParent())
    
    logging.info("MNI registration complete.\n")   
            
        

#
# Autoelectrodes
#

class Autoelectrodes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Auto electrodes"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["SEEG"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Justo Montoya (Pompeu Fabra University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Autoelectrodes">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Justo Montoya and was partially funded by XXXXXXX grant XXXXXXXXXXXX.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Autoelectrodes1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Autoelectrodes',
        sampleName='Autoelectrodes1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'Autoelectrodes1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='Autoelectrodes1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='Autoelectrodes1'
    )

    # Autoelectrodes2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Autoelectrodes',
        sampleName='Autoelectrodes2',
        thumbnailFileName=os.path.join(iconsPath, 'Autoelectrodes2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='Autoelectrodes2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='Autoelectrodes2'
    )


#
# AutoelectrodesWidget
#

class AutoelectrodesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Autoelectrodes.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AutoelectrodesLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
            # firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            # if firstVolumeNode:
            #     self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        # self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        # self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        # self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        # self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        # if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
        #     self.ui.applyButton.toolTip = "Compute output volume"
        #     self.ui.applyButton.enabled = True
        # else:
        #     self.ui.applyButton.toolTip = "Select input and output volume nodes"
        #     self.ui.applyButton.enabled = False
        
        self.ui.applyButton.toolTip = "Define electrodes"
        self.ui.applyButton.enabled = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process()



#
# AutoelectrodesLogic
#

class AutoelectrodesLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "100.0")
        # if not parameterNode.GetParameter("Invert"):
        #     parameterNode.SetParameter("Invert", "false")

    def process(self):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        # if not inputVolume or not outputVolume:
        #     raise ValueError("Input or output volume is invalid")

        # import time
        # startTime = time.time()
        # logging.info('Processing started')
        
        # # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        # cliParams = {
        #     'InputVolume': inputVolume.GetID(),
        #     'OutputVolume': outputVolume.GetID(),
        #     'ThresholdValue': imageThreshold,
        #     'ThresholdType': 'Above' if invert else 'Below'
        # }
        # cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        # slicer.mrmlScene.RemoveNode(cliNode)

        # stopTime = time.time()
        # logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

        logging.info("Starting...")
        regionsMNI()
        findContacts()


#
# AutoelectrodesTest
#

class AutoelectrodesTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Autoelectrodes1()

    def test_Autoelectrodes1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        # import SampleData
        # registerSampleData()
        # inputVolume = SampleData.downloadSample('Autoelectrodes1')
        self.delayDisplay('Loaded test data set')

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # Test the module logic

        logic = AutoelectrodesLogic()

        # Test algorithm with non-inverted threshold
        logic.process()
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
