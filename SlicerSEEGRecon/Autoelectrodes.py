

from scipy import signal

import json

import logging
import os

import vtk
import ctk, DICOMLib

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

from NiBabelModelIO import NiBabelModelIOLogic

import re
import numpy as np
import csv
from itertools import compress
from scipy.ndimage import binary_dilation

import os
from os import listdir
from os.path import isfile,join

import time

import warnings

import qt

try:
    import pandas as pd
except: 
    import pip
    pip_install("pandas")
    import pandas as pd
       
try:
    import plotly
except: 
    import pip
    pip_install("plotly")
    import plotly

try: 
    from PySide2 import QtWidgets
except:
    import pip
    pip_install("PySide2")
    from PySide2 import QtWidgets




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
    fidNode.GetNthControlPointPosition(n,pos)
    return pos

def computeBrainVolume(brainModelNode):
    
    mesh = brainModelNode.GetMesh()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    
    # Obtain the volume in cm^3
    brain_volume = mass.GetVolume()/1000 # The /1000 is to convert from mm^3 which is the default
    
    return brain_volume

def natural_keys(text):
    """
    Sort helper: splits strings into list of ints and strings for natural sort.
    E.g., "B10" -> ["B", 10]
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]

def findContacts(fidNode):
    
    # Get the aseg map
    global asegVolumeNode, asegVoxelArray
    # In the new versions of the module the aseg volume is called brain_segmentation but aseg in the old ones
    asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('brain_segmentation')
    if not asegVolumeNode: 
        asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('aseg')
    asegVoxelArray = slicer.util.arrayFromVolume(asegVolumeNode)
    
    # All markup's names and positions in RAS coordinates
    markup_names = [fidNode.GetNthControlPointLabel(i) for i in range(fidNode.GetNumberOfControlPoints())]
    markup_RAS = [NthFiducialPosition(fidNode,i) for i in range(fidNode.GetNumberOfControlPoints())]
    
    # Remove markups that signal the END of the electrode (wich is not the last contact but the tip of the elctrode, outside of the skull)
    boolean = [has_numbers(name) for name in markup_names]
    clean_markup_names = list(compress(markup_names, boolean))
    clean_markup_RAS = list(compress(markup_RAS, boolean))
    
    # Sort the lists alphabetically
    tuples = zip(*sorted(zip(clean_markup_names, clean_markup_RAS)))
    mu, RAS = [list(tuple) for tuple in  tuples]
    markups = ["{}{:0>2.0f}".format(''.join([i for i in markup if not i.isdigit()]),int(re.search(r'\d+', markup).group()))  for markup in mu]
    markups = zip(*sorted(zip(markups,RAS), key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)))
    markups, RAS = [list(tuple) for tuple in  markups]
    # print(markups)
    
    # WARN THE USER IF THE ELECTRODES DO NOT MATCH
    test = [''.join([i for i in markup if not i.isdigit()]) for markup in markups]
    if any([counts < 2 for counts in [test.count(electrode) for electrode in np.unique(test)]]):
        unique_count = [test.count(electrode) for electrode in np.unique(test)]
        bool_count = [count < 2 for count in unique_count]
        bad_electrodes = list(compress(np.unique(test), bool_count))
        # popup message
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setText("Warning")
        msg.setInformativeText("Electrodes tips and ends are not matching")
        msg.setWindowTitle("Warning!")
        msg.setDetailedText('Issue with: {}'.format(bad_electrodes))
        msg.setStandardButtons(qt.QMessageBox.Ok | qt.QMessageBox.Cancel)
        retval = msg.exec_()
    
    # Store also the location of the end of the electrodes just for representational purposes
    end_boolean = [not element for element in boolean]
    end_markup_names = list(compress(markup_names, end_boolean))
    end_markup_RAS = list(compress(markup_RAS, end_boolean))
    end_tuples = zip(*sorted(zip(end_markup_names, end_markup_RAS)))
    # end_markups, end_RAS = [list(tuple) for tuple in  end_tuples]
    
    
    #############################################################################
    # MONOPOLAR: The tool just adds the contacts where they really are in space.#
    #############################################################################
    # Goal: Compute the position of the remaining contacts and include them in the monopolar markup list (real-R/L)
    
    # Initialize lists for the markups and their RAS coordinates
    global monopolar_markups, monopolar_RAS
    monopolar_markups = []
    monopolar_RAS = []
    
    # Initialize the variables where the WHITE matter nodes will be stored
    monopolar_markups_White_Matter = []
    monopolar_RAS_White_Matter = []
    fidNode_White_Matter = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_White_Matter.SetName(fidNode.GetName()+"_White_Matter")
    slicer.mrmlScene.AddNode(fidNode_White_Matter)

    # Initialize the variables where the GREY matter nodes will be stored
    monopolar_markups_Grey_Matter = []
    monopolar_RAS_Grey_Matter = []
    fidNode_Grey_Matter = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_Grey_Matter.SetName(fidNode.GetName()+"_Grey_Matter")
    slicer.mrmlScene.AddNode(fidNode_Grey_Matter)
    
    # Initialize the variables where the nodes OUTSIDE of the brain will be stored
    monopolar_markups_outside_brain = []
    monopolar_RAS_outside_brain = []
    fidNode_outside_brain = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_outside_brain.SetName(fidNode.GetName()+"_outside_brain")
    slicer.mrmlScene.AddNode(fidNode_outside_brain)
    
    # Initialize the variables where the END nodes will be stored (nodes that refer to the end of the electrode shaft, outside the skull)
    monopolar_markups_E = []
    monopolar_RAS_E = []
    fidNodeE = slicer.vtkMRMLMarkupsFiducialNode()
    fidNodeE.SetName(fidNode.GetName()+"_ends")
    slicer.mrmlScene.AddNode(fidNodeE)
    
    # Initialize variables to store all the electrode lines in the same folder
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode() # Get hierarychyNode
    SceneID = shNode.GetSceneItemID() # Get Current SceneID
    electrodes_folderID = shNode.CreateFolderItem(SceneID, "Electrodes") # Create folder
    
    # Iterate over all the markups defined by the user
    line_done_names = []
    for index,markup in enumerate(markups): 
        if index < len(markups)-1: # There's a -1 one here because the penultimate markup adds the last markup, so there is no need to check the last one
        
            # Check the letter/name and the number of the selected markup and the next one in the list
            letter = ''.join([i for i in markup if not i.isdigit()])
            digit = int(re.search(r'\d+', markup).group())
            next_letter = ''.join([i for i in markups[index+1] if not i.isdigit()])
            next_digit = int(re.search(r'\d+', markups[index+1]).group())
            
            # If they have the same letter we can continue because it means that are part of the same electrode
            if letter == next_letter:
                
                # In case that the node was not initially defined, add initial contact to the new list
                if markup not in monopolar_markups:
                    monopolar_markups.append(markup)
                    monopolar_RAS.append(RAS[index])
                    
                    # IJK and real anatomic location of that 1st contact (White Matter/Grey Matter/Outside)
                    ijk_position = RAStoIJK(RAS[index], asegVolumeNode)
                    anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                    
                    # Store the node in the corresponding volume
                    if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                        monopolar_markups_White_Matter.append(markup)
                        monopolar_RAS_White_Matter.append(RAS[index])
                        fidNode_White_Matter.AddControlPoint(RAS[index], letter+str(digit))
                    elif "Unknown" in anatomic_position:
                        monopolar_markups_outside_brain.append(markup)
                        monopolar_RAS_outside_brain.append(RAS[index])
                        fidNode_outside_brain.AddControlPoint(RAS[index], letter+str(digit))
                    else:
                        monopolar_markups_Grey_Matter.append(markup)
                        monopolar_RAS_Grey_Matter.append(RAS[index])
                        fidNode_Grey_Matter.AddControlPoint(RAS[index], letter+str(digit))
                        
                # Calculate how many markups should be added until the next user-defined markup
                additions = next_digit - digit-1
                
                # Calculate how distant the markups should be
                distance = np.subtract(RAS[index+1], RAS[index])/(additions+1)
                
                # Add these new markups
                for i in range(additions):
                    
                    new_digit = digit+i+1
                    new_position = np.add(RAS[index], distance*(i+1))
                    monopolar_markups.append(letter+str(new_digit))
                    monopolar_RAS.append(new_position)
                    
                    fidNode.AddControlPoint(new_position, letter+str(new_digit))
                    
                    # IJK and real anatomic location of the rest of the contacts (White Matter/Grey Matter/Outside), but not the last one
                    ijk_position = RAStoIJK(new_position, asegVolumeNode)
                    anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                    
                    # Store the node in the corresponding volume
                    if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                        monopolar_markups_White_Matter.append(letter+str(new_digit))
                        monopolar_RAS_White_Matter.append(new_position)
                        fidNode_White_Matter.AddControlPoint(new_position, letter+str(new_digit))
                    elif "Unknown" in anatomic_position:
                        monopolar_markups_outside_brain.append(letter+str(new_digit))
                        monopolar_RAS_outside_brain.append(new_position)
                        fidNode_outside_brain.AddControlPoint(new_position, letter+str(new_digit))
                    else:
                        monopolar_markups_Grey_Matter.append(letter+str(new_digit))
                        monopolar_RAS_Grey_Matter.append(new_position)
                        fidNode_Grey_Matter.AddControlPoint(new_position, letter+str(new_digit))
                
                # At this point we have reached the last node of the electrode, we can directly add it
                monopolar_markups.append(letter+str(additions+2)) # Here the name of the node is additions+2 to account for the 1st one and last one (this) 
                monopolar_RAS.append(RAS[index+1])
                
                # IJK and real anatomic location of the last contact (White Matter/Grey Matter/Outside)
                ijk_position = RAStoIJK(RAS[index+1], asegVolumeNode)
                anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                
                # Store the node in the corresponding volume
                if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                    monopolar_markups_White_Matter.append(letter+str(additions+2))   
                    monopolar_RAS_White_Matter.append(RAS[index+1])
                    fidNode_White_Matter.AddControlPoint(RAS[index+1], letter+str(additions+2))
                elif "Unknown" in anatomic_position:
                    monopolar_markups_outside_brain.append(letter+str(additions+2))
                    monopolar_RAS_outside_brain.append(RAS[index+1])
                    fidNode_outside_brain.AddControlPoint(RAS[index+1], letter+str(additions+2))
                else:
                    monopolar_markups_Grey_Matter.append(letter+str(additions+2))
                    monopolar_RAS_Grey_Matter.append(RAS[index+1])
                    fidNode_Grey_Matter.AddControlPoint(RAS[index+1], letter+str(additions+2))
                
                #Create LineNode to represent the whole electrode up to the last contact
                lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
                
                # Check if that electrode has been already done or not
                lineName = letter
                if lineName not in line_done_names: 
                    line_done_names.append(lineName)
                    lineNode.SetName(letter+"_segment_1")
                else:
                    i=0
                    while lineName in line_done_names:
                        i +=1
                        lineName = '%s_%i' % (letter, i)
                    line_done_names.append(lineName)
                    lineNode.SetName(letter+"_segment_"+str(i+1))
                
                # Create the actual line in the LineNode
                line_coordinates = np.array([RAS[index],RAS[index+1]])
                slicer.util.updateMarkupsControlPointsFromArray(lineNode, line_coordinates)
                
                # Change color of the line to blue/red according to right/left hemisphere
                if RAS[index][0] > 0:
                    lineNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
                    lineNode.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
                else:
                    lineNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
                    lineNode.GetDisplayNode().SetSelectedColor([170/255,0/255,0/255])
                
                # Hide text
                lineNode.GetDisplayNode().SetTextScale(0)
                # Set glyph scale (reduce almost to none to make it not visible but the line still can be shown)
                lineNode.GetDisplayNode().SetGlyphScale(0.1)
                # Set line thickness
                lineNode.GetDisplayNode().SetLineThickness(15)
                # Lock the line
                lineNode.SetLocked(True)
                
                # Store the line in the electrodes folder
                lineNodeID = shNode.GetItemByDataNode(lineNode)
                shNode.SetItemParent(lineNodeID, electrodes_folderID)
            
            # If the letter is not the same as the next one it means we found the last markup of the electrode, thus we can extend this last section to better graphically represent the electrode
            else:
                
                # Compute the vector that defines the line that passes through the last point and the penultimate user-defined markup
                pointA = RAS[index-1]
                pointB = RAS[index]
                l = pointB[0]-pointA[0]
                m = pointB[1]-pointA[1]
                n = pointB[2]-pointA[2]
                AB = np.array([l,m,n])
                
                # Select the last point
                pointE = pointE = [(abs(coordinate)+30)*np.sign(coordinate) for coordinate in pointB]
                l = pointE[0]-pointA[0]
                m = pointE[1]-pointA[1]
                n = pointE[2]-pointA[2]
                AE = np.array([l,m,n])
                
                # Project the last markup (E) onto the line generated by AB
                P = pointA + np.dot(AE,AB) / np.dot(AB,AB) * AB
                
                # Generate projection on Slicer 
                fidNodeE.AddControlPoint(P,letter)
                
                #Create LineNode to represent the whole electrode up to the last contact
                lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
                
                # Check if that electrode has been already done or not
                lineName = letter
                if lineName not in line_done_names: 
                    line_done_names.append(lineName)
                    lineNode.SetName(letter+"_segment_1")
                else:
                    i=0
                    while lineName in line_done_names:
                        i +=1
                        lineName = '%s_%i' % (letter, i)
                    line_done_names.append(lineName)
                    lineNode.SetName(letter+"_segment_"+str(i+1))
                
                # Create the actual line in the LineNode
                line_coordinates = np.array([RAS[index],P])
                slicer.util.updateMarkupsControlPointsFromArray(lineNode, line_coordinates)
                
                # Change color of the line to blue/red according to right/left hemisphere
                if P[0] > 0:
                    lineNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
                    lineNode.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
                else:
                    lineNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
                    lineNode.GetDisplayNode().SetSelectedColor([170/255,0/255,0/255])
                    
                # Hide text
                lineNode.GetDisplayNode().SetTextScale(0)
                # Set glyph scale (reduce almost to none to make it not visible but the line still can be shown)
                lineNode.GetDisplayNode().SetGlyphScale(0.1)
                # Set line thickness
                lineNode.GetDisplayNode().SetLineThickness(15)
                # Lock the line
                lineNode.SetLocked(True)
                
                # Store the line in the electrodes folder
                lineNodeID = shNode.GetItemByDataNode(lineNode)
                shNode.SetItemParent(lineNodeID, electrodes_folderID)
        else:
            
            # Compute the vector that defines the line that passes through the last point and the penultimate user-defined markup
            pointA = RAS[index-1]
            pointB = RAS[index]
            l = pointB[0]-pointA[0]
            m = pointB[1]-pointA[1]
            n = pointB[2]-pointA[2]
            AB = np.array([l,m,n])
            
            # Select the last point
            pointE = pointE = [(abs(coordinate)+30)*np.sign(coordinate) for coordinate in pointB]
            l = pointE[0]-pointA[0]
            m = pointE[1]-pointA[1]
            n = pointE[2]-pointA[2]
            AE = np.array([l,m,n])
            
            # Project the last markup (E) onto the line generated by AB
            P = pointA + np.dot(AE,AB) / np.dot(AB,AB) * AB
            
            # Generate projection on Slicer 
            fidNodeE.AddControlPoint(P,letter)
            
            #Create LineNode to represent the whole electrode up to the last contact
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            
            # Check if that electrode has been already done or not
            lineName = letter
            if lineName not in line_done_names: 
                line_done_names.append(lineName)
                lineNode.SetName(letter+"_segment_1")
            else:
                i=0
                while lineName in line_done_names:
                    i +=1
                    lineName = '%s_%i' % (letter, i)
                line_done_names.append(lineName)
                lineNode.SetName(letter+"_segment_"+str(i+1))
                
            # Create the actual line in the LineNode
            line_coordinates = np.array([RAS[index],P])
            slicer.util.updateMarkupsControlPointsFromArray(lineNode, line_coordinates)
            
            # Change color of the line to blue/red according to right/left hemisphere
            if P[0] > 0:
                lineNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
                lineNode.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
            else:
                lineNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
                lineNode.GetDisplayNode().SetSelectedColor([170/255,0/255,0/255])
            
            # Hide text
            lineNode.GetDisplayNode().SetTextScale(0)
            # Set glyph scale (reduce almost to none to make it not visible but the line still can be shown)
            lineNode.GetDisplayNode().SetGlyphScale(0.1)
            # Set line thickness
            lineNode.GetDisplayNode().SetLineThickness(15)
            # Lock the line
            lineNode.SetLocked(True)
            
            # Store the line in the electrodes folder
            lineNodeID = shNode.GetItemByDataNode(lineNode)
            shNode.SetItemParent(lineNodeID, electrodes_folderID)
            
    # Count the number of contacts on White Matter
    monopolar_number_nodes_White_Matter = len(monopolar_markups_White_Matter)
    monopolar_number_nodes_White_Matter_Right = len([1 for RAScoord in monopolar_RAS_White_Matter if RAScoord[0] > 0])
    monopolar_number_nodes_White_Matter_Left = monopolar_number_nodes_White_Matter - monopolar_number_nodes_White_Matter_Right
    
    # Count the number of contacts on Grey Matter
    monopolar_number_nodes_Grey_Matter = len(monopolar_markups_Grey_Matter)
    monopolar_number_nodes_Grey_Matter_Right = len([1 for RAScoord in monopolar_RAS_Grey_Matter if RAScoord[0] > 0])
    monopolar_number_nodes_Grey_Matter_Left = monopolar_number_nodes_Grey_Matter - monopolar_number_nodes_Grey_Matter_Right
    
    # Count the number of contacts outside brain
    monopolar_number_nodes_outside_brain = len(monopolar_markups_outside_brain)
    monopolar_number_nodes_outside_brain_Right = len([1 for RAScoord in monopolar_RAS_outside_brain if RAScoord[0] > 0])
    monopolar_number_nodes_outside_brain_Left = monopolar_number_nodes_outside_brain - monopolar_number_nodes_outside_brain_Right

    # Obtain volume of the hemispheres 
    right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rh_grey')
    if not right_hemisphere_VolumeNode: 
        right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rhp')
    right_hemisphere_Volume = computeBrainVolume(right_hemisphere_VolumeNode)
    left_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('lh_grey')
    if not left_hemisphere_VolumeNode: 
        left_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('lhp')
    left_hemisphere_Volume = computeBrainVolume(left_hemisphere_VolumeNode)
    brain_Volume = right_hemisphere_Volume + left_hemisphere_Volume
    
    # Dictionary to store all the values
    global monopolar_counted_nodes_Dict
    monopolar_counted_nodes_Dict = {"WhiteMatter_Right": monopolar_number_nodes_White_Matter_Right,
                                    "WhiteMatter_Left": monopolar_number_nodes_White_Matter_Left,
                                    "WhiteMatter": monopolar_number_nodes_White_Matter,
                                    "GreyMatter_Right": monopolar_number_nodes_Grey_Matter_Right,
                                    "GreyMatter_Left": monopolar_number_nodes_Grey_Matter_Left,
                                    "GreyMatter": monopolar_number_nodes_Grey_Matter,
                                    "Outside_Right": monopolar_number_nodes_outside_brain_Right,
                                    "Outside_Left": monopolar_number_nodes_outside_brain_Left,
                                    "Outside": monopolar_number_nodes_outside_brain,
                                    "BrainVolume_Right": right_hemisphere_Volume,
                                    "BrainVolume_Left": left_hemisphere_Volume,
                                    "BrainVolume": brain_Volume}
        
    # Lock and select all markups
    for markupN in range(fidNode.GetNumberOfControlPoints()):
        fidNode.SetNthControlPointLocked(markupN,True) # Lock 
        fidNode.SetNthControlPointSelected(markupN,1) # Select
    for markupN in range(fidNode_White_Matter.GetNumberOfControlPoints()):
        fidNode_White_Matter.SetNthControlPointLocked(markupN,True)
        fidNode_White_Matter.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNode_Grey_Matter.GetNumberOfControlPoints()):
        fidNode_Grey_Matter.SetNthControlPointLocked(markupN,True)
        fidNode_Grey_Matter.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNode_outside_brain.GetNumberOfControlPoints()):
        fidNode_outside_brain.SetNthControlPointLocked(markupN,True)
        fidNode_outside_brain.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNodeE.GetNumberOfControlPoints()):
        fidNodeE.SetNthControlPointLocked(markupN,True)
        fidNodeE.SetNthControlPointSelected(markupN,1)
    
    if P[0] > 0:
        # SELECTED Color for the right hemisphere
        fidNode.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNode_White_Matter.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNode_Grey_Matter.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNodeE.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        # UNselected Color for the right hemisphere
        fidNode.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNode_White_Matter.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNode_Grey_Matter.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNodeE.GetDisplayNode().SetColor([0/255,0/255,225/255])
    else:
        fidNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNode_White_Matter.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNode_Grey_Matter.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNodeE.GetDisplayNode().SetColor([170/255,0/255,0/255])
    
    # Gliph type
    fidNode.GetDisplayNode().SetGlyphType(8)
    fidNode.GetDisplayNode().SetGlyphScale(5)
    fidNode.GetDisplayNode().SetTextScale(0)
    
    fidNode_White_Matter.GetDisplayNode().SetGlyphType(8)
    fidNode_White_Matter.GetDisplayNode().SetGlyphScale(3)
    fidNode_White_Matter.GetDisplayNode().SetTextScale(0)
    fidNode_White_Matter.GetDisplayNode().SetVisibility(False)
    
    fidNode_Grey_Matter.GetDisplayNode().SetGlyphType(8)
    fidNode_Grey_Matter.GetDisplayNode().SetGlyphScale(3)
    fidNode_Grey_Matter.GetDisplayNode().SetTextScale(0)
    fidNode_White_Matter.GetDisplayNode().SetVisibility(False)
    
    fidNodeE.GetDisplayNode().SetGlyphType(8)
    fidNodeE.GetDisplayNode().SetGlyphScale(3)
    fidNodeE.GetDisplayNode().SetTextScale(0)
    fidNode_White_Matter.GetDisplayNode().SetVisibility(False)
    
    fidNode.GetDisplayNode().SetVisibility(True)
    
    logging.info("Monopolar contact placement complete.\n") 
   
    
    ###########
    # BIPOLAR #
    ###########
        
    # Initialize the variables where the BIPOLAR referenced nodes will be stored (This location is not real, just a representacion of where the re-referenced source would be)
    fidNode_Bipolar = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_Bipolar.SetName("Bipolar_"+fidNode.GetName())
    slicer.mrmlScene.AddNode(fidNode_Bipolar)
    bipolar_markups = []
    bipolar_RAS = []
    
    # Initialize the variables where the BIPOLAR WHITE matter nodes will be stored
    fidNode_Bipolar_White_Matter = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_Bipolar_White_Matter.SetName("Bipolar_"+fidNode.GetName()+"_White_Matter")
    slicer.mrmlScene.AddNode(fidNode_Bipolar_White_Matter)
    bipolar_markups_White_Matter = []
    bipolar_RAS_White_Matter = []
    
    # Initialize the variables where the BIPOLAR GREY matter nodes will be stored
    fidNode_Bipolar_Grey_Matter = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_Bipolar_Grey_Matter.SetName("Bipolar_"+fidNode.GetName()+"_Grey_Matter")
    slicer.mrmlScene.AddNode(fidNode_Bipolar_Grey_Matter)
    bipolar_markups_Grey_Matter = []
    bipolar_RAS_Grey_Matter = []
    
    # Initialize the variables where the BIPOLAR nodes OUTSIDE of the brain will be stored
    fidNode_Bipolar_outside_brain = slicer.vtkMRMLMarkupsFiducialNode()
    fidNode_Bipolar_outside_brain.SetName("Bipolar_"+fidNode.GetName()+"_Outside")
    slicer.mrmlScene.AddNode(fidNode_Bipolar_outside_brain)
    bipolar_markups_outside_brain = []
    bipolar_RAS_outside_brain = []
    
    for index,markup in enumerate(monopolar_markups):
        if index < len(monopolar_markups)-1:
            
            letter = ''.join([i for i in markup if not i.isdigit()])
            digit = int(re.search(r'\d+', markup).group())
            next_letter = ''.join([i for i in monopolar_markups[index+1] if not i.isdigit()])
            next_digit = int(re.search(r'\d+', monopolar_markups[index+1]).group())
                            
            if letter == next_letter:
                middle_point = np.add(monopolar_RAS[index+1], monopolar_RAS[index])/2
                bi_tag = "-".join([letter+str(digit),monopolar_markups[index+1]])
                bipolar_markups.append(bi_tag)
                bipolar_RAS.append(middle_point)
                
                fidNode_Bipolar.AddControlPoint(middle_point, bi_tag)
                
                # IJK and real anatomic location of the contact (White Matter/Grey Matter/Outside)
                ijk_position = RAStoIJK(middle_point, asegVolumeNode)
                anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                
                # Store the node in the corresponding volume
                if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                    bipolar_markups_White_Matter.append(bi_tag)
                    bipolar_RAS_White_Matter.append(middle_point)
                    fidNode_Bipolar_White_Matter.AddControlPoint(middle_point, bi_tag)
                elif "Unknown" in anatomic_position:
                    bipolar_markups_outside_brain.append(bi_tag)
                    bipolar_RAS_outside_brain.append(middle_point)
                    fidNode_Bipolar_outside_brain.AddControlPoint(middle_point, bi_tag)
                else:
                    bipolar_markups_Grey_Matter.append(bi_tag)
                    bipolar_RAS_Grey_Matter.append(middle_point)
                    fidNode_Bipolar_Grey_Matter.AddControlPoint(middle_point, bi_tag)
                
    # Lock all markups
    for markupN in range(fidNode_Bipolar.GetNumberOfControlPoints()):
        fidNode_Bipolar.SetNthControlPointLocked(markupN,True)
        fidNode_Bipolar.SetNthControlPointSelected(markupN,0)
    for markupN in range(fidNode_Bipolar_White_Matter.GetNumberOfControlPoints()):
        fidNode_Bipolar_White_Matter.SetNthControlPointLocked(markupN,True)
        fidNode_Bipolar_White_Matter.SetNthControlPointSelected(markupN,0)
    for markupN in range(fidNode_Bipolar_Grey_Matter.GetNumberOfControlPoints()):
        fidNode_Bipolar_Grey_Matter.SetNthControlPointLocked(markupN,True)
        fidNode_Bipolar_Grey_Matter.SetNthControlPointSelected(markupN,0)
    
    if P[0] > 0:
        # SELECTED color for the right hemisphere
        fidNode_Bipolar.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        fidNode_Bipolar_White_Matter.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        # UNselected
        fidNode_Bipolar.GetDisplayNode().SetColor([0/255,0/255,0/255])
        fidNode_Bipolar_White_Matter.GetDisplayNode().SetColor([0/255,0/255,0/255])
        fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetColor([0/255,0/255,0/255])
        
    else:
        # Selected
        fidNode_Bipolar.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        fidNode_Bipolar_White_Matter.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
        # Unselected
        fidNode_Bipolar.GetDisplayNode().SetColor([0/255,0/255,0/255])
        fidNode_Bipolar_White_Matter.GetDisplayNode().SetColor([0/255,0/255,0/255])
        fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetColor([0/255,0/255,0/255])
    
    # Gliph type
    fidNode_Bipolar.GetDisplayNode().SetGlyphType(8)
    fidNode_Bipolar.GetDisplayNode().SetGlyphScale(3)
    fidNode_Bipolar.GetDisplayNode().SetTextScale(0)
    
    fidNode_Bipolar_White_Matter.GetDisplayNode().SetGlyphType(8)
    fidNode_Bipolar_White_Matter.GetDisplayNode().SetGlyphScale(3)
    fidNode_Bipolar_White_Matter.GetDisplayNode().SetTextScale(0)
    
    fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetGlyphType(8)
    fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetGlyphScale(3)
    fidNode_Bipolar_Grey_Matter.GetDisplayNode().SetTextScale(0)
    
    # Count the number of contacts on White Matter
    bipolar_number_nodes_White_Matter = len(bipolar_markups_White_Matter)
    bipolar_number_nodes_White_Matter_Right = len([1 for RAScoord in bipolar_RAS_White_Matter if RAScoord[0] > 0])
    bipolar_number_nodes_White_Matter_Left = bipolar_number_nodes_White_Matter - bipolar_number_nodes_White_Matter_Right
    
    # Count the number of contacts on Grey Matter
    bipolar_number_nodes_Grey_Matter = len(bipolar_markups_Grey_Matter)
    bipolar_number_nodes_Grey_Matter_Right = len([1 for RAScoord in bipolar_RAS_Grey_Matter if RAScoord[0] > 0])
    bipolar_number_nodes_Grey_Matter_Left = bipolar_number_nodes_Grey_Matter - bipolar_number_nodes_Grey_Matter_Right
    
    # Count the number of contacts outside brain
    bipolar_number_nodes_outside_brain = len(bipolar_markups_outside_brain)
    bipolar_number_nodes_outside_brain_Right = len([1 for RAScoord in bipolar_RAS_outside_brain if RAScoord[0] > 0])
    bipolar_number_nodes_outside_brain_Left = bipolar_number_nodes_outside_brain - bipolar_number_nodes_outside_brain_Right
    
    # Dictionary to store all the values
    global bipolar_counted_nodes_Dict
    bipolar_counted_nodes_Dict = {"WhiteMatter_Right": bipolar_number_nodes_White_Matter_Right,
                                    "WhiteMatter_Left": bipolar_number_nodes_White_Matter_Left,
                                    "WhiteMatter": bipolar_number_nodes_White_Matter,
                                    "GreyMatter_Right": bipolar_number_nodes_Grey_Matter_Right,
                                    "GreyMatter_Left": bipolar_number_nodes_Grey_Matter_Left,
                                    "GreyMatter": bipolar_number_nodes_Grey_Matter,
                                    "Outside_Right": bipolar_number_nodes_outside_brain_Right,
                                    "Outside_Left": bipolar_number_nodes_outside_brain_Left,
                                    "Outside": bipolar_number_nodes_outside_brain,
                                    "BrainVolume_Right": right_hemisphere_Volume,
                                    "BrainVolume_Left": left_hemisphere_Volume,
                                    "BrainVolume": brain_Volume}
    
    logging.info("Bipolar contact placement complete.\n") 
    
    
    # If the markup file is old, it will be loaded as a fcsv and will fail to save as mrb the whole file.
    # To solve it, the markup fiducial node is re-done, making it mrk.json 
    newfidNode = slicer.vtkMRMLMarkupsFiducialNode()
    newfidNode.SetName(fidNode.GetName())
    slicer.mrmlScene.AddNode(newfidNode)
    
    for name,RAS in zip(monopolar_markups, monopolar_RAS):
        newfidNode.AddControlPoint(RAS, name)
    
    # Remove og node
    slicer.mrmlScene.RemoveNode(fidNode)
        
def registerMNI(fixedVolumeNode):
    global mniPath, linearTransformNode
    
    # ICBM152  
    # Paths
    mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
    templatePath = os.path.join(mniPath, 'icbm_avg_152_t1_tal_lin.nii') # moving volume
    movingmaskPath = os.path.join(mniPath, 'icbm_avg_152_t1_tal_lin_mask.nii') # moving volume mask
    
    # Set parameters
    movingVolumeNode = slicer.util.loadVolume(templatePath,properties={"name":"ICBM152_T1","center":False})
    
    linearTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    linearTransformNode.SetName("Transform2MNI")
    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolumeNode.SetName("ICBM152_registered")
    
    aseg = slicer.util.getFirstNodeByName("aseg")
    fixedmaskNode = slicer.mrmlScene.CopyNode(aseg)
    fixedmaskNode.SetName("aseg_mask")
    movingmaskNode = slicer.util.loadVolume(movingmaskPath,properties={"name":"MNI_mask","labelmap":True,"center":False})
    
    parameters = {}
    parameters["fixedVolume"] = fixedVolumeNode
    parameters["movingVolume"] = movingVolumeNode
    parameters["samplingPercentage"] = 0.02
    parameters["linearTransform"] = linearTransformNode
    parameters["outputVolume"] = outputVolumeNode
    parameters["initializeTransformMode"] = "useCenterOfHeadAlign"
    # parameters["useRigid"] = True
    # parameters["useScaleVersor3D"] = True
    # parameters["useScaleSkewVersor3D"] = True
    parameters["useAffine"] = True
    parameters["maskProcessingMode"] = "ROI"
    parameters["fixedBinaryVolume"] = fixedmaskNode
    parameters["movingBinaryVolume"] = movingmaskNode
    parameters["outputFixedVolumeROI"] = fixedmaskNode
    parameters["outputMovingVolumeROI"] = fixedmaskNode
    
    # Execution
    generalRegistration = slicer.modules.brainsfit
    cliNode = slicer.cli.run(generalRegistration, None, parameters)
    
    logging.info("MNI registration complete.\n")   
            
def regionsMNI(destinyDirectory):
    
    # path of the label mapfile
    labelmapPath = os.path.join(mniPath, 'mni_icbm152_CerebrA_tal_nlin_sym_09c.nii') # labelmap
    
    # Select the transform from the MNI to patient registration 
    transform = slicer.util.getFirstNodeByName("Transform2MNI")
    
    # Transform the labelmap to match the patient volume
    labelmapVolumeNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_labels","labelmap":True,"center":False})
    labelmapVolumeNode.SetName("transformed_MNI_labels")
    # labelmapVolumeNode.ApplyTransformMatrix(transform.GetMatrixTransformToParent())
    labelmapVolumeNode.SetAndObserveTransformNodeID(transform.GetID())
    time.sleep(15)
    slicer.util.forceRenderAllViews()
    time.sleep(5)
    
def regionsMNI_2(destinyDirectory):  
    
    # Select the transform from the MNI to patient registration 
    transform = slicer.util.getFirstNodeByName("Transform2MNI")
    
    labelmapVolumeNode = slicer.util.getFirstNodeByName("transformed_MNI_labels")
    
    # Obtain voxel array of the label map to obtain the number associated to a specific location
    MNIVoxelArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    
    # Load MNI table relating number tag to area
    MNI_details = pd.read_csv(os.path.join(mniPath, 'CerebrA_LabelDetails.csv'))
    
    # Atlases
    # Initialize dataframe for the atlases
    atlas = pd.DataFrame(columns=['Contact', 'Aseg', 'MNI'])
    
    # Inverse transform to compute the MNI coordinates of each contact
    worldToMniTransform = vtk.vtkGeneralTransform()
    transform.GetTransformFromWorld(worldToMniTransform)
    
    # Fill dataframe
    for index,contact in enumerate(monopolar_markups):
        
        #  transform ras to mni
        ras = monopolar_RAS[index]
        mni = [0,0,0]
        worldToMniTransform.TransformPoint(ras, mni)
        
        # Obtain Aseg Labels
        point_ijk = RAStoIJK(ras,asegVolumeNode)
        aseg_label = anatomicREL(asegVoxelArray[point_ijk[2],point_ijk[1],point_ijk[0]])
        
        # Obtain MNI Labels
        point_ijk = RAStoIJK(ras,labelmapVolumeNode)
        mni_label_number = MNIVoxelArray[point_ijk[2],point_ijk[1],point_ijk[0]]
        
        # In case that the label is non-existent. check surroundings
        surround_index = 1
        while mni_label_number == 0 and surround_index<5:
            surroundings = [-surround_index,0,surround_index]
            areas = []
            for x in surroundings:
                for y in surroundings:
                    for z in surroundings:
                        try:
                            areas.append(MNIVoxelArray[point_ijk[2]+x,point_ijk[1]+y,point_ijk[0]+z])
                        except:
                            pass
            mni_label_number = max(set(areas), key = areas.count)
            surround_index = surround_index+1
        
        if mni_label_number != 0:
            mni_label = MNI_details[MNI_details.eq(mni_label_number).any(axis="columns")]["Label Name"].iloc[0]
        else:
            mni_label = "unknown"
        
        # print(contact)
        # print(ras)
        # print(mni)
        # print(point_ijk)
        # print(mni_label_number)
        # print(mni_label)
        
        
        # Fill dataframe
        df = pd.DataFrame([[contact, aseg_label, mni_label, mni[0], mni[1], mni[2]]], columns=['Contact', 'Aseg', 'MNI', 'X_mni', 'Y_mni', 'Z_mni'])
        atlas = pd.concat([atlas, df])

    # Save the files
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        atlas.to_csv(path_or_buf=destinyDirectory+"/electrodes.csv", index=False, index_label=False)
        print_atlas = atlas.to_string(index=False)
        print(print_atlas)

def regionsMNInibabel(fixedVolumeNode):
    
    brain_img_data = arrayFromVolume(fixedVolumeNode)

def importAndLoadDICOMFolder(dicom_folder):
    # Step 1: Create and initialize DICOM database
    dbDir = os.path.join(dicom_folder, "MyDICOMDb")
    os.makedirs(dbDir, exist_ok=True)
    dbPath = os.path.join(dbDir, "ctkDICOM.sql")

    db = ctk.ctkDICOMDatabase()
    db.openDatabase(dbPath, "SLICER")
    slicer.dicomDatabase = db  # attach to Slicer

    # Step 2: Import DICOM folder
    indexer = ctk.ctkDICOMIndexer()
    indexer.addDirectory(db, dicom_folder)
    indexer.waitForImportFinished()

    # Step 3: Load all series
    patients = db.patients()
    for patientUID in patients:
        studies = db.studiesForPatient(patientUID)
        for studyUID in studies:
            series = db.seriesForStudy(studyUID)
            for seriesUID in series:
                DICOMLib.loadSeriesByUID([seriesUID])
        
#
# Autoelectrodes
#

class Autoelectrodes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Autoelectrodes"  # TODO: make this more human readable by adding spaces
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
        """Traceback (most recent call last):
  File "/home/marzieh/Dropbox/M/my-plotly/SlicerAutoelectrodes-main/SlicerAutoelectrodes/Autoelectrodes/Autoelectrodes.py", line 1636, in openEDFviewer
    popup = PlotWindow()
NameError: name 'PlotWindow' is not defined
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
        # self.ui = slicer.util.childWidgetVariables(uiWidget)
        

        def print_all_children(widget, level=0):
            try:
                name = widget.objectName
            except:
                name = 'NO NAME'
            print("  " * level + f"{name} ({type(widget).__name__})")
            for child in widget.children():
                print_all_children(child, level + 1)

        # print("🔍 Full UI tree:")
        # print_all_children(uiWidget)

        print("🔍 UI children:")
        # print(uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.applyButton) #.children()
        

        # # Verify widget loading
        # if not hasattr(self.ui, 'DirectoryButton_saveBundle'):
        #     logging.error("Failed to load DirectoryButton_saveBundle from UI file")
        #     return

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AutoelectrodesLogic()

        # Initialize parameter node first
        self.initializeParameterNode()
        
        # Add observers for scene events
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)


        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.inputSelector = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.inputSelector
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.electrodesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.electrodesSelector = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.electrodesSelector
        self.electrodesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.fixedVolumeSelector = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.fixedVolumeSelector
        self.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.inputListSelector = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.inputListSelector
        self.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.outputListSelector = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.outputListSelector
        self.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.checkBox_transfer = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.transferCollapsibleButton.checkBox_transfer
        self.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.DirectoryButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.DirectoryButton
        self.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.DirectoryButton_subject = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.DirectoryButton_subject
        self.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_saveBundle.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.DirectoryButton_saveBundle = uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.DirectoryButton_saveBundle
        self.DirectoryButton_saveBundle.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_rawFolder.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.DirectoryButton_rawFolder = uiWidget.TabWidget.qt_tabwidget_stackedwidget.importTab.ImportGroupBox.DirectoryButton_rawFolder
        self.DirectoryButton_rawFolder.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        
        # Other inputs
        # self.ui.saveFileName.editingFinished.connect(self.updateParameterNodeFromGUI)
        self.saveFileName = uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.saveFileName
        self.saveFileName.connect('editingFinished()', self.updateParameterNodeFromGUI)

        # Visualization advanced inputs
        # self.ui.templateName.activated.connect(self.updateParameterNodeFromGUI)
        self.templateName = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.visualizationCollapsibleButton.templateName
        self.templateName.connect('activated(QString)', self.updateParameterNodeFromGUI)    
        # self.ui.applySettingsButton.connect('clicked(bool)', self.onApplyTemplateSettingsButton)
        self.applySettingsButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.visualizationCollapsibleButton.applySettingsButton
        self.applySettingsButton.connect('clicked(bool)', self.onApplyTemplateSettingsButton)
         
        # Buttons
        self.applyButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.applyButton
        self.applyButton.connect('clicked(bool)', self.onApplyButton)

        # self.ui.pushButton.connect('clicked(bool)', self.onPushButton)
        self.pushButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.pushButton
        self.pushButton.connect('clicked(bool)', self.onPushButton)
        # self.ui.copyButton.connect('clicked(bool)', self.onCopyButton)
        self.copyButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.copyButton
        self.copyButton.connect('clicked(bool)', self.onCopyButton)
        # self.ui.pushButton_mapping.connect('clicked(bool)', self.onPushButton_mapping)
        self.pushButton_mapping = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.pushButton_mapping
        self.pushButton_mapping.connect('clicked(bool)', self.onPushButton_mapping) 
        # self.ui.saveTableButton.connect('clicked(bool)', self.onSaveTableButton)
        self.saveTableButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.saveTableButton
        self.saveTableButton.connect('clicked(bool)', self.onSaveTableButton)
        # self.ui.saveBundleButton.connect('clicked(bool)', self.onSaveBundleButton)
        self.saveBundleButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.saveBundleButton
        self.saveBundleButton.connect('clicked(bool)', self.onSaveBundleButton)
        # self.ui.ImportPatientButton.connect('clicked(bool)', self.onImportPatientButton)
        self.ImportPatientButton = uiWidget.TabWidget.qt_tabwidget_stackedwidget.importTab.ImportGroupBox.ImportPatientButton
        self.ImportPatientButton.connect('clicked(bool)', self.onImportPatientButton)   

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

        self._parameterNode = self.logic.getParameterNode()
        if not self._parameterNode.GetParameter("saveDirectory"):
            self._parameterNode.SetParameter("saveDirectory", "")

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByName("real")
        #     if firstVolumeNode:
        #         self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

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
        self.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.electrodesSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputElectrodes"))
        self.fixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("fixedVolume"))
        self.checkBox_transfer.checked = (self._parameterNode.GetParameter("Transfer") == "false")
        self.inputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputListVolume"))
        self.outputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputListVolume"))
        self.DirectoryButton.directory = str(self._parameterNode.GetParameter("Directory"))
        self.DirectoryButton_subject.directory = str(self._parameterNode.GetParameter("Directory_Subject"))
        self.DirectoryButton_saveBundle.directory = str(self._parameterNode.GetParameter("saveDirectory"))
        self.DirectoryButton_rawFolder.directory = str(self._parameterNode.GetParameter("rawDirectory"))
        self.saveFileName.text = str(self._parameterNode.GetParameter("saveFileName"))
                
        self.visulizeMarkupsWidget.setCurrentNode(self._parameterNode.GetNodeReference("visualizationInputVolume"))
        self.templateName.setCurrentText(str(self._parameterNode.GetParameter("templateName")))

        # Update buttons states and tooltips (Enable)
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.applyButton.toolTip = "Generate electrodes"
            self.applyButton.enabled = True
        else:
            self.applyButton.toolTip = "Select input volume node"
            self.applyButton.enabled = False
            
        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputElectrodes"):
            self.saveTableButton.toolTip = "Generate table"
            self.saveTableButton.enabled = True
        else:
            self.saveTableButton.toolTip = "Select input"
            self.saveTableButton.enabled = False
        
        if self._parameterNode.GetParameter("saveDirectory"):
            self.saveBundleButton.enabled = True
        else:
            self.saveBundleButton.enabled = False

        if self._parameterNode.GetParameter("rawDirectory"):
            self.ImportPatientButton.enabled = True
        else:
            self.ImportPatientButton.enabled = False
        
        
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

        self._parameterNode.SetNodeReferenceID("InputVolume", self.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputElectrodes", self.electrodesSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("fixedVolume", self.fixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputListVolume", self.inputListSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputListVolume", self.outputListSelector.currentNodeID)
        self._parameterNode.SetParameter("Directory", str(self.DirectoryButton.directory))
        self._parameterNode.SetParameter("Directory_Subject", str(self.DirectoryButton_subject.directory))
        self._parameterNode.SetParameter("saveDirectory", str(self.DirectoryButton_saveBundle.directory))
        self._parameterNode.SetParameter("rawDirectory", str(self.DirectoryButton_rawFolder.directory))
        self._parameterNode.SetParameter("Transfer", "true" if self.checkBox_transfer.checked else "false")
        self._parameterNode.SetParameter("saveFileName", str(self.saveFileName.text))
                
        # self._parameterNode.SetNodeReferenceID("visualizationInputVolume", self.ui.visulizeMarkupsWidget.currentNode().currentNodeID)
        self._parameterNode.SetParameter("templateName", str(self.templateName.currentText))

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fidNode = self.inputSelector.currentNode()
            
            # checked_mapping = self.ui.checkBox_mapping.checked
            # fixedVolumeNode = self.ui.fixedVolumeSelector.currentNode()
            
            self.logic.process(fidNode)
    
    def onPushButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fixedVolumeNode = self.fixedVolumeSelector.currentNode()
            self.logic.regions(fixedVolumeNode)
            
            destinyDirectory = self.DirectoryButton.directory
            self.logic.regions_parttwo(destinyDirectory)
    
    def onApplyTemplateSettingsButton(self):
        
        electrodeList = self.visulizeMarkupsWidget.currentNode()
        
        templateName = self.templateName.currentText
        
        self.logic.applyTemplate(templateName)
    
    def onPushButton_mapping(self):
        
        destinyDirectory = self.DirectoryButton.directory
        self.logic.regions_partthree(destinyDirectory)
        
            
    def onSaveTableButton(self):
        
        destinyDirectory = self.DirectoryButton_subject.directory
        electrodesNode = self.electrodesSelector.currentNode()        
        
        self.logic.save_info(destinyDirectory, electrodesNode)
        
    def onSaveBundleButton(self):
        
        destinyDirectory = self.DirectoryButton_saveBundle.directory
        name_of_file = self.saveFileName.text
                
        self.logic.save_Bundle(destinyDirectory, name_of_file)
    
    def onImportPatientButton(self):

        rawDirectory = self.DirectoryButton_rawFolder.directory
        print(f"Importing patient data from: {rawDirectory}")

        self.logic.import_patient(rawDirectory)

    def onCopyButton(self):
        
        inputList = self.inputListSelector.currentNode()
        outputList = self.outputListSelector.currentNode()
        checked_transfer = self.checkBox_transfer.checked
        
        self.logic.copy_transfer(inputList, outputList, checked_transfer)
        

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
        if not parameterNode.GetParameter("Bipolar"):
            parameterNode.SetParameter("Bipolar", "false")

    def process(self, fidNode):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param fidNode: Markup node where the start and ending contacts of the electrodes to be processed are
        :param fixedVolumeNode: Volume node to be used as fixed volume for the MNI registration of the brain
        
        """

        if not fidNode:
            raise ValueError("Input volume or fixed volume is invalid")

        startTime = time.time()
        logging.info("Processing electrodes...")
        
        findContacts(fidNode)
        
        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    

    def update_all_markups(self):

        # Get all fiducial markup nodes
        markupNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")

        for markupNode in markupNodes:
            numPoints = markupNode.GetNumberOfControlPoints()
            if numPoints == 0:
                continue

            # --- Determine color based on average X position ---
            x_coords = []
            for i in range(numPoints):
                pos = [0.0, 0.0, 0.0]
                markupNode.GetNthControlPointPositionWorld(i, pos)
                x_coords.append(pos[0])
            mean_x = sum(x_coords) / len(x_coords)
            color = (1.0, 0.0, 0.0) if mean_x < 0 else (0.0, 0.0, 1.0)  # Red or Blue

            # Set display properties
            displayNode = markupNode.GetDisplayNode()
            if displayNode:
                displayNode.SetGlyphTypeFromString("Circle2D")
                displayNode.SetSelectedColor(color)
                displayNode.SetColor(color)

            # Lock the markup node
            markupNode.SetLocked(True)

            # --- Reorder control points using natural sort on label ---
            controlPoints = []
            for i in range(numPoints):
                pos = [0.0, 0.0, 0.0]
                markupNode.GetNthControlPointPositionWorld(i, pos)
                label = markupNode.GetNthControlPointLabel(i)
                controlPoints.append((label, pos))

            # Sort using natural order
            controlPoints.sort(key=lambda x: natural_keys(x[0]))

            # Remove and re-add in new order
            markupNode.RemoveAllControlPoints()
            for label, pos in controlPoints:
                markupNode.AddControlPointWorld(*pos)
                markupNode.SetNthControlPointLabel(markupNode.GetNumberOfControlPoints() - 1, label)
    
    def update_all_models(self):
    
        # Define color mapping based on keywords
        color_map = {
            "white": (1.0, 1.0, 1.0),               # White
            "grey": (0.839, 0.839, 0.839),          # Silver (#d6d6d6)
            "hyp": (0.0, 0.992, 1.0),               # Cyan (#00FDFF)
            "amy": (1.0, 0.984, 0.0)                # Yellow (#FFFB00)
        }

        # Go through all model nodes
        modelNodes = slicer.util.getNodesByClass("vtkMRMLModelNode")

        for modelNode in modelNodes:
            name = modelNode.GetName().lower()
            displayNode = modelNode.GetDisplayNode()

            if not displayNode:
                continue

            # --- Set color ---
            for key, rgb in color_map.items():
                if key in name:
                    displayNode.SetColor(rgb)
                    displayNode.SetSelectedColor(rgb)
                    break  # Use the first match found

            # --- Set opacity ---
            displayNode.SetOpacity(0.15)

            # --- Set visibility ---
            if any(key in name for key in ["grey", "hyp", "amy"]):
                displayNode.SetVisibility(True)
            else:
                displayNode.SetVisibility(False)


    def applyTemplate(self, templateName):
        
        """
        Apply a template to the selected markups node.
        :param templateName: Name of the template to apply
        """

        # Apply the template
        if templateName == "EpiDB":
            self.update_all_markups()
            self.update_all_models()

        elif templateName == "Template2":
            # Apply Template2 settings
            pass
        else:
            logging.error(f"Unknown template: {templateName}")
    
    def regions(self,fixedVolumeNode):
        
        logging.info("Mapping electrodes into the MNI space...")
        registerMNI(fixedVolumeNode)
    
    def regions_parttwo(self,destinyDirectory):
        
        # time.sleep(15)
        regionsMNI(destinyDirectory)
        
    def regions_partthree(self,destinyDirectory):
        regionsMNI_2(destinyDirectory)
        
    def copy_transfer(self,inputList,outputList,transfer):
                
        # All markup's names and positions in RAS coordinates
        markup_names = [inputList.GetNthControlPointLabel(i) for i in range(inputList.GetNumberOfControlPoints())]
        markup_RAS = [NthFiducialPosition(inputList,i) for i in range(inputList.GetNumberOfControlPoints())]
        markup_selected = [inputList.GetNthControlPointSelected(i) for i in range(inputList.GetNumberOfControlPoints())]
        
        # Just the selected 
        selected_markup_names = list(compress(markup_names, markup_selected))
        selected_markup_RAS = list(compress(markup_RAS, markup_selected))
        
        for i, markup_name in enumerate(selected_markup_names):
            # First copy the electrodes in the new list
            outputList.AddControlPoint(selected_markup_RAS[i], markup_name)
            outputList.SetNthControlPointLocked(i,True)
            outputList.SetNthControlPointSelected(i,0)
            # If transfer is selected, remove those electrodes from the original list
            if transfer:
                available_markup_names = [inputList.GetNthControlPointLabel(i) for i in range(inputList.GetNumberOfControlPoints())]
                #  This loop is needed because each time that an electrode is removed from the markup list, it updates, so the index oof the remaining electrodes change
                for index, og_markup_name in enumerate(available_markup_names):
                    if markup_name == og_markup_name:
                        inputList.RemoveNthControlPoint(index)
    
    def save_info(self,destinationDirectory, fidNode):
        
        """ This function saves information of the contacts of the electrodes 
        """
        
        global asegVolumeNode, asegVoxelArray
        asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('brain_segmentation')
        if not asegVolumeNode: 
            asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('aseg')
        asegVoxelArray = slicer.util.arrayFromVolume(asegVolumeNode)
        
        # All markup's names and positions in RAS coordinates
        markup_names = [fidNode.GetNthControlPointLabel(i) for i in range(fidNode.GetNumberOfControlPoints())]
        markup_RAS = [NthFiducialPosition(fidNode,i) for i in range(fidNode.GetNumberOfControlPoints())]
        
        # Remove markups that signal the END of the electrode (wich is not the last contact but the tip of the elctrode, outside of the skull)
        boolean = [has_numbers(name) for name in markup_names]
        clean_markup_names = list(compress(markup_names, boolean))
        clean_markup_RAS = list(compress(markup_RAS, boolean))
        
        # Sort the lists alphabetically
        tuples = zip(*sorted(zip(clean_markup_names, clean_markup_RAS)))
        mu, RAS = [list(tuple) for tuple in  tuples]
        markups = ["{}{:0>2.0f}".format(''.join([i for i in markup if not i.isdigit()]),int(re.search(r'\d+', markup).group()))  for markup in mu]
        markups = zip(*sorted(zip(markups,RAS), key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item)))
        markups, RAS = [list(tuple) for tuple in  markups]
        
        # Initialize the variables where the WHITE matter nodes will be stored
        markups_White_Matter = []
        RAS_White_Matter = []
        
        # Initialize the variables where the GREY matter nodes will be stored
        markups_Grey_Matter = []
        RAS_Grey_Matter = []
        
        # Initialize the variables where the nodes OUTSIDE of the brain will be stored
        markups_outside_brain = []
        RAS_outside_brain = []
        
        for index,markup in enumerate(markups): 
            
            # IJK and real anatomic location of that 1st contact (White Matter/Grey Matter/Outside)
            ijk_position = RAStoIJK(RAS[index], asegVolumeNode)
            anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
            
            # Store the node in the corresponding volume
            if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                markups_White_Matter.append(markup)
                RAS_White_Matter.append(RAS[index])
            elif "Unknown" in anatomic_position:
                markups_outside_brain.append(markup)
                RAS_outside_brain.append(RAS[index])
            else:
                markups_Grey_Matter.append(markup)
                RAS_Grey_Matter.append(RAS[index])
                
            
        # Count the number of contacts on White Matter
        number_nodes_White_Matter = len(markups_White_Matter)
        number_nodes_White_Matter_Right = len([1 for RAScoord in RAS_White_Matter if RAScoord[0] > 0])
        number_nodes_White_Matter_Left = number_nodes_White_Matter - number_nodes_White_Matter_Right
        
        # Count the number of contacts on Grey Matter
        number_nodes_Grey_Matter = len(markups_Grey_Matter)
        number_nodes_Grey_Matter_Right = len([1 for RAScoord in RAS_Grey_Matter if RAScoord[0] > 0])
        number_nodes_Grey_Matter_Left = number_nodes_Grey_Matter - number_nodes_Grey_Matter_Right
        
        # Count the number of contacts outside brain
        number_nodes_outside_brain = len(markups_outside_brain)
        number_nodes_outside_brain_Right = len([1 for RAScoord in RAS_outside_brain if RAScoord[0] > 0])
        number_nodes_outside_brain_Left = number_nodes_outside_brain - number_nodes_outside_brain_Right

        # Obtain volume of the hemispheres 
        right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rh_grey')
        if not right_hemisphere_VolumeNode: 
            right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rhp')
        right_hemisphere_Volume = computeBrainVolume(right_hemisphere_VolumeNode)
        left_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('lh_grey')
        if not left_hemisphere_VolumeNode:
            left_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('lhp')
        left_hemisphere_Volume = computeBrainVolume(left_hemisphere_VolumeNode)
        brain_Volume = right_hemisphere_Volume + left_hemisphere_Volume
        
        # Dictionary to store all the values
        global monopolar_counted_nodes_Dict
        monopolar_counted_nodes_Dict = {"WhiteMatter_Right": number_nodes_White_Matter_Right,
                                        "WhiteMatter_Left": number_nodes_White_Matter_Left,
                                        "WhiteMatter": number_nodes_White_Matter,
                                        "GreyMatter_Right": number_nodes_Grey_Matter_Right,
                                        "GreyMatter_Left": number_nodes_Grey_Matter_Left,
                                        "GreyMatter": number_nodes_Grey_Matter,
                                        "Outside_Right": number_nodes_outside_brain_Right,
                                        "Outside_Left": number_nodes_outside_brain_Left,
                                        "Outside": number_nodes_outside_brain,
                                        "BrainVolume_Right": right_hemisphere_Volume,
                                        "BrainVolume_Left": left_hemisphere_Volume,
                                        "BrainVolume": brain_Volume}
            
        try:
            # Save dictionary of  nodes
            with open(destinationDirectory+"/"+"electrodes_count_"+fidNode.GetName()+".csv", "w", newline="") as f:
                w = csv.DictWriter(f, monopolar_counted_nodes_Dict.keys())
                w.writeheader()
                w.writerow(monopolar_counted_nodes_Dict)
            
            print("Electrode placement info saved!")
                
        except:
            logging.error("Saving failed")

    def registerCTtoT1MRI(self, t1VolumeNode, ctVolumeNode):

        """        Register CT volume to T1 MRI volume.
        :param t1VolumeNode: T1 MRI volume node
        :param ctVolumeNode: CT volume node
        
        Following: Registration of brain CT images to an MRI template for the purpose of lesion-symptom mapping, Hugo J. Kuijf 
        https://link.springer.com/chapter/10.1007/978-3-319-02126-3_12#citeas
        """

        volumesLogic = slicer.modules.volumes.logic()
        transformLogic = slicer.modules.transforms.logic()

        print("     Extracting bone from CT volume...")

        # Extract the bone from the CT volume
        ctVolume = volumesLogic.CloneVolume(slicer.mrmlScene,ctVolumeNode, 'CT_Bones')
        volumesLogic.CenterVolume(ctVolume)
        boneThreshold = 500 
        ctArray = slicer.util.arrayFromVolume(ctVolume)
        ctArray[ctArray <= boneThreshold] = 0  # Set values below threshold to 0
        slicer.util.arrayFromVolumeModified(ctVolume)

        print("     Loading template...")
        # Extract bone from the MNI152 template
        resPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
        mniPath = os.path.join(resPath, 'icbm_avg_152_t1_tal_lin.nii')
        mniBrainMaskPath = os.path.join(resPath, 'icbm_avg_152_t1_tal_lin_mask.nii')
        mniHeadMaskPath = os.path.join(resPath, 'icbm_avg_152_t1_tal_lin_homemadeHeadMask.nii')

        mniVolume = slicer.util.loadVolume(mniPath,properties={"name":"ICBM152","center":True})
        
        mniBrainMaskVolume = slicer.util.loadVolume(mniBrainMaskPath, properties={"name": "ICBM152_BrainMask", "center": True})
        mniBrainMaskArray = slicer.util.arrayFromVolume(mniBrainMaskVolume).copy()
        mniBrainMaskArray[mniBrainMaskArray < 1] = 0 # Ensure binary mask
        mniBrainMaskArray[mniBrainMaskArray >= 1] = 1  

        mniHeadMaskVolume = slicer.util.loadVolume(mniHeadMaskPath, properties={"name": "ICBM152_HeadMask", "labelmap":True, "center": True})
        mniHeadMaskArray = slicer.util.arrayFromVolume(mniHeadMaskVolume).copy()
        mniHeadMaskArray[mniHeadMaskArray < 1] = 0 # Ensure binary mask
        mniHeadMaskArray[mniHeadMaskArray >= 1] = 1

        mniBonesVolume = volumesLogic.CloneVolume(slicer.mrmlScene, mniVolume, 'ICBM152_Bones')
        mniBonesArray = slicer.util.arrayFromVolume(mniBonesVolume)
        v_min = np.min(mniBonesArray)
        v_max = np.max(mniBonesArray)
        mniBonesArray = v_max - (mniBonesArray - v_min) # Invert the values to get bone density
        mniBonesArray[mniBonesArray <= 250000] = 0  # Set values below threshold to 0 (remove soft tissue)
        mniBonesArray[mniBrainMaskArray == 1] = 0  # Set values inside the brain mask to 0 (remove the brain)
        mniBonesArray[mniHeadMaskArray != 1] = 0  # Set values outside the head mask to 0 (remove the tissue outside of skull and background)

        slicer.util.updateVolumeFromArray(mniBonesVolume, mniBonesArray)

        print("     Registering Template to CT... WAIT ...")
        outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'ICBM_CTregistered')
        ICBMtoCT_Transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'ICBMtoCT')

        slicer.util.selectModule("Elastix")
        elastix_widget = slicer.modules.elastix.widgetRepresentation().self()

        elastix_widget.ui.fixedVolumeSelector.setCurrentNode(ctVolume)
        elastix_widget.ui.movingVolumeSelector.setCurrentNode(mniVolume)
        elastix_widget.ui.outputVolumeSelector.setCurrentNode(outputVolume)
        elastix_widget.ui.outputTransformSelector.setCurrentNode(ICBMtoCT_Transform)

        registrationPreset = elastix_widget.logic.getRegistrationPresets()[0] # Default preset (general transform)
        elastix_widget._parameterNode.SetParameter(elastix_widget.logic.REGISTRATION_PRESET_ID_PARAM, registrationPreset.getID())

        elastix_widget.onApplyButton()

        # Invert the transform to apply it to the CT volume later
        CTtoICBM_Transform = slicer.mrmlScene.CopyNode(ICBMtoCT_Transform)
        CTtoICBM_Transform.Inverse()

        print("     Registering Template to T1 MRI... WAIT ...")
        mniVolume = slicer.util.getNode("ICBM152")
        outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'ICBM_MRIregistered')
        ICBMtoMRI_Transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'ICBMtoMRI')
        
        elastix_widget.ui.fixedVolumeSelector.setCurrentNode(t1VolumeNode)
        elastix_widget.ui.movingVolumeSelector.setCurrentNode(mniVolume)
        elastix_widget.ui.outputVolumeSelector.setCurrentNode(outputVolume)
        elastix_widget.ui.outputTransformSelector.setCurrentNode(ICBMtoMRI_Transform)

        registrationPreset = elastix_widget.logic.getRegistrationPresets()[0] # Default preset (general transform)
        elastix_widget._parameterNode.SetParameter(elastix_widget.logic.REGISTRATION_PRESET_ID_PARAM, registrationPreset.getID())

        elastix_widget.onApplyButton()

        print("     Applying non-linear transforms to CT volume...")
        # Apply the transforms to the CT volume
        ctVolumeNode = slicer.util.getFirstNodeByName("CT_postElectrodes_RAW")
        ctRegisteredVolumeNode = slicer.mrmlScene.CopyNode(ctVolumeNode)
        ctRegisteredVolumeNode.SetName("CT_Registered_ICBM")
        ctRegisteredVolumeNode.SetAndObserveTransformNodeID(CTtoICBM_Transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(ctRegisteredVolumeNode)

        ctRegisteredVolumeNode = slicer.util.getFirstNodeByName("CT_Registered_ICBM")
        ctRegisteredVolumeNode = slicer.mrmlScene.CopyNode(ctRegisteredVolumeNode)
        ctRegisteredVolumeNode.SetName("CT_Registered_ICBM_GOAL")
        ctRegisteredVolumeNode.SetAndObserveTransformNodeID(ICBMtoMRI_Transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(ctRegisteredVolumeNode)

        print("     Registering Goal to CT... WAIT ...")
        # Register the goal to the CT volume
        ctGoalVolumeNode = slicer.util.getFirstNodeByName("CT_Registered_ICBM_GOAL")
        ctRawVolumeNode = slicer.util.getFirstNodeByName("CT_postElectrodes_RAW")
        outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'CT_postElectrodes')
        CTGoaltoCT_Transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'CTGoaltoCT')

        elastix_widget.ui.fixedVolumeSelector.setCurrentNode(ctRawVolumeNode)
        elastix_widget.ui.movingVolumeSelector.setCurrentNode(ctGoalVolumeNode)
        elastix_widget.ui.outputVolumeSelector.setCurrentNode(outputVolume)
        elastix_widget.ui.outputTransformSelector.setCurrentNode(CTGoaltoCT_Transform)

        registrationPreset = elastix_widget.logic.getRegistrationPresets()[1] # RIGID TRANSFORM
        elastix_widget._parameterNode.SetParameter(elastix_widget.logic.REGISTRATION_PRESET_ID_PARAM, registrationPreset.getID())

        elastix_widget.onApplyButton()

        # Invert the transform to apply it to the CT 
        CTtoGoal_Transform = slicer.mrmlScene.CopyNode(CTGoaltoCT_Transform)
        CTtoGoal_Transform.Inverse()

        print("     Applying transform to CT volume...")
        # Apply the transform to the CT volume
        ctRawVolumeNode = slicer.util.getFirstNodeByName("CT_postElectrodes_RAW")
        ctFinalVolumeNode = slicer.mrmlScene.CopyNode(ctRawVolumeNode)
        ctFinalVolumeNode.SetName("CT_postElectrodes_Final")
        ctFinalVolumeNode.SetAndObserveTransformNodeID(CTtoGoal_Transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(ctFinalVolumeNode)

        slicer.util.selectModule("Autoelectrodes")

        print("     Done!")


    def import_patient(self, rawDirectory):

        """
        Import patient data from a raw folder.
        :param rawDirectory: Directory containing the raw patient data
        """

        # Count time it takes to import the patient data
        startTime = time.time()

        # Search for specific files in all subfolders of rawDirectory
        file_names_to_search = [
            "aseg.mgz", "norm.mgz", "T1.mgz",
            "rh.pial", "rh.white", "lh.pial", "lh.white"]
        found_files = {}

        dicom_folder = None
        for root, dirs, files in os.walk(rawDirectory):
            # Check for DICOM folder (contains .dcm files)
            if any(f.lower().endswith('.dcm') for f in files):
                dicom_folder = root  # Save the folder path containing DICOM files

            for file_name in file_names_to_search:
                if file_name in files and file_name not in found_files:
                    found_files[file_name] = os.path.join(root, file_name)# Search for aseg.mgz in all subfolders of rawDirectory
        
        print('Loading DICOM data... \n')
        
        # Load DICOM data
        importAndLoadDICOMFolder(dicom_folder)

        # IMPORTANT: Since the DICOM is the only vtkMRMLScalarVolumeNode we look for it and change the name (it cannot be changed in the importAndLoadDICOMFolder function)
        # This should be always the first volume loaded, so it is safe to do this
        dicomVolumeNode = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")[0]
        if dicomVolumeNode:
            dicomVolumeNode.SetName("CT_postElectrodes_RAW")

            volumesLogic = slicer.modules.volumes.logic()
            volumesLogic.CenterVolume(dicomVolumeNode)

        for filename in found_files:

            print(f"Processing file: {filename}")

            file_path = found_files[filename]

            # Load the segmentation file
            if filename.endswith('.mgz') and 'aseg' in filename.lower():
                print(f"Loading segmentation: {filename} ...\n")
                
                # Load aseg as labelmap volume
                labelmapNode = slicer.util.loadVolume(file_path, {'labelmap': True})
                labelmapNode.SetName("aseg")

                # Set color table to FreeSurfer labels
                fsColorNode = slicer.util.getNode('FreeSurferLabels')
                labelmapNode.GetDisplayNode().SetAndObserveColorNodeID(fsColorNode.GetID())

                # # Optional: show the segmentation
                # segmentationNode.CreateDefaultDisplayNodes()
                # segmentationNode.GetDisplayNode().SetVisibility2DFill(True)

            if filename.endswith('.mgz') and ('norm' in filename.lower() or 'T1' in filename):
                print(f"Loading volume: {filename} ...\n")
                volumeNode = slicer.util.loadVolume(file_path)
                if 'norm' in filename.lower():
                    volumeNode.SetName("brain")
                elif 'T1' in filename:
                    volumeNode.SetName('MRI_T1')  # Use filename without extension as node name
                    volumeNode.GetDisplayNode().SetVisibility(True)
            
            # Load cortical surfaces
            if filename.endswith('.pial') or filename.endswith('.white'):
                print(f"Loading surface: {filename} ...\n")
                parts = filename.split('.')
                hemi, surfaceType = parts  
                baseName = f"{hemi}_{surfaceType}"

                # Read file content
                logic = NiBabelModelIOLogic()   
                loadedNode = logic.createAndReadModelNode(file_path, baseName)

        print("All files processed successfully!\n")

        print("Now registering the CT volume to the T1 MRI volume...\n")

        # # Now co-register the T1 volume (fixed) with the CT volume (moving)
        t1VolumeNode = slicer.util.getNode("MRI_T1")
        ctVolumeNode = slicer.util.getNode("CT_postElectrodes_RAW")

        self.registerCTtoT1MRI(t1VolumeNode, ctVolumeNode)

        print("Files imported successfully!\n")

        # Count time it took to import the patient data
        stopTime = time.time()
        elapsed_minutes = int((stopTime - startTime) // 60)
        elapsed_seconds = int((stopTime - startTime) % 60)
        print(f'Importing patient data completed in {elapsed_minutes} min {elapsed_seconds} sec')


        # if t1VolumeNode and ctVolumeNode:
        #     linearTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        #     linearTransformNode.SetName("TransformCTtoT1")
        #     outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        #     outputVolumeNode.SetName("CT_postElectrodes") 

        #     parameters = {"fixedVolume": t1VolumeNode,
        #                   "movingVolume": ctVolumeNode,
        #                   "samplingPercentage": 0.02,
        #                   "linearTransform": linearTransformNode,
        #                   "outputVolume": outputVolumeNode,
        #                   "initializeTransformMode": "useCenterOfHeadAlign",
        #                   "useRigid": True}
            
        #     # Execution
        #     generalRegistration = slicer.modules.brainsfit
        #     cliNode = slicer.cli.run(generalRegistration, None, parameters)



    def save_Bundle(self, destinationDirectory, name_of_file):
        
        # Create folder of the patient
        destinationDirectory = os.path.join(destinationDirectory, name_of_file)

        namesOfFilesToSearch = ['aseg',
                                '3D',
                                'brain',
                                'norm',
                                'lhp',
                                'lh_pial',
                                'lh_grey',
                                'rhp',
                                'rh_pial',
                                'rh_grey',
                                'lhw',
                                'lh_white',
                                'rhw',
                                'rh_white',
                                'pet.3d',
                                'pet.3D',
                                'PET.3D',
                                'PET.3d',
                                'ct.3d',
                                'CT.3d',
                                'ct.3D',
                                'CT.3D',
                                'ctp.3d',
                                'CTp.3d',
                                'ctp.3D',
                                'CTp.3D',
                                'res',
                                'resection',
                                'real-L','real-R',
                                'real-L-P','real-R-P',
                                'electrodes_L','electrodes_R',
                                'electrodes_L_Grey_Matter','electrodes_R_Grey_Matter',
                                'Bipolar-real-L','Bipolar-real-R',
                                'Bipolar-real-L-P','Bipolar-real-R-P',
                                'Bipolar_electrodes_L','Bipolar_electrodes_R',
                                'Bipolar_electrodes_L_Grey_Matter','Bipolar_electrodes_R_Grey_Matter',
                                'etc','ETC']
        
        nrrdFiles = ['aseg','3D', 'brain', 'norm',
                     'pet.3d','pet.3D','PET.3D','PET.3d',
                     'ct.3d','CT.3d','ct.3D','CT.3D',
                     'ctp.3d','CTp.3d','ctp.3D','CTp.3D',]
        
        vtkFiles = ['lhp','lh_pial', 'lh_grey',
                    'rhp','rh_pial', 'rh_grey',
                    'lhw','lh_white',
                    'rhw','rh_white',
                    'res','resection']
        
        mrkJsonFiles = ['real-L','real-R',
                        'electrodes_L','electrodes_R',
                        'real-L-P','real-R-P',
                        'electrodes_L_Grey_Matter','electrodes_R_Grey_Matter',
                        'Bipolar-real-L','Bipolar-real-R',
                        'Bipolar-real-L-P','Bipolar-real-R-P',
                        'Bipolar_electrodes_L','Bipolar_electrodes_R',
                        'Bipolar_electrodes_L_Grey_Matter','Bipolar_electrodes_R_Grey_Matter',
                        'etc','ETC']
        
        renamingDict = {'aseg':'brain_segmentation',
                        '3D': 'MRI_T1',
                        'brain':'brain',
                        'norm':'brain',
                        'lhp':'lh_grey',
                        'lh_pial':'lh_grey',
                        'lh_grey':'lh_grey',
                        'rhp':'rh_grey',
                        'rh_pial':'rh_grey',
                        'rh_grey':'rh_grey',
                        'lhw':'lh_white',
                        'lh_white':'lh_white',
                        'rhw':'rh_white',
                        'rh_white':'rh_white',
                        'pet.3d':'PET',
                        'pet.3D':'PET',
                        'PET.3D':'PET',
                        'PET.3d':'PET',
                        'ct.3d':'CT',
                        'CT.3d':'CT',
                        'ct.3D':'CT',
                        'CT.3D':'CT',
                        'ctp.3d':'CT_postElectrodes',
                        'CTp.3d':'CT_postElectrodes',
                        'ctp.3D':'CT_postElectrodes',
                        'CTp.3D':'CT_postElectrodes',
                        'ctps.3d':'CT_postElectrodes',
                        'CTps.3d':'CT_postElectrodes',
                        'ctps.3D':'CT_postElectrodes',
                        'CTps.3D':'CT_postElectrodes',
                        'res':'resection',
                        'resection':'resection',
                        'real-L':'electrodes_L',
                        'real-R':'electrodes_R',
                        'real-L-P':'electrodes_L_grey',
                        'real-R-P':'electrodes_L_grey',
                        'electrodes_L':'electrodes_L',
                        'electrodes_R':'electrodes_R',
                        'electrodes_L_Grey_Matter':'electrodes_L_grey',
                        'electrodes_R_Grey_Matter':'electrodes_R_grey',
                        'Bipolar-real-L':'electrodes_bipolar_L',
                        'Bipolar-real-R':'electrodes_bipolar_R',
                        'Bipolar-real-L-P':'electrodes_bipolar_L_grey',
                        'Bipolar-real-R-P':'electrodes_bipolar_R_grey',
                        'Bipolar_electrodes_L':'electrodes_bipolar_L',
                        'Bipolar_electrodes_R':'electrodes_bipolar_R',
                        'Bipolar_electrodes_L_Grey_Matter':'electrodes_bipolar_L_grey',
                        'Bipolar_electrodes_R_Grey_Matter':'electrodes_bipolar_R_grey',
                        'etc':'ETC',
                        'ETC':'ETC'}

        # Make sure that the transforms are hardened before saving
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode")
        for volumeNode in volumeNodes:
            transformNode = volumeNode.GetParentTransformNode()
            if transformNode:
                print(f"Harden transform for volume: {volumeNode.GetName()}")
                slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)
        
        # Save the files
        for nameOfFile in namesOfFilesToSearch:
            print(f"\nSearching for: {nameOfFile}")

            if slicer.util.getFirstNodeByName(nameOfFile):
                # Locate the nodes in the scene
                selectedNode = slicer.util.getFirstNodeByName(nameOfFile)
                print(f"Found node: {selectedNode.GetName()} ")
                
                # Select new name
                newFileName = renamingDict[nameOfFile]

                # Select file format
                if nameOfFile in nrrdFiles:
                    file_extension = '.nrrd'
                    print(f"File format: {file_extension}")
                if nameOfFile in vtkFiles:
                    file_extension = '.vtk'
                    print(f"File format: {file_extension}")
                if nameOfFile in mrkJsonFiles:
                    file_extension = '.mrk.json'
                    print(f"File format: {file_extension}")

                # Stablish file path
                if (file_extension == '.nrrd') or (file_extension == '.vtk'):
                    new_file_path = os.path.normpath(os.path.join(destinationDirectory, 'anatomy', f"{newFileName}{file_extension}"))
                if file_extension == '.mrk.json':
                    new_file_path = os.path.normpath(os.path.join(destinationDirectory, 'electrodes', f"{newFileName}{file_extension}"))

                # Save the node with the new name
                if slicer.util.saveNode(selectedNode, new_file_path):
                    print(f"'{selectedNode.GetName()}' saved successfully as {new_file_path}")
                else:
                    print(f"Failed to save '{selectedNode.GetName()}' as {new_file_path}")

            else:
                pass
            

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
        
        testPath = '/Volumes/GoogleDrive/Mi unidad/_PhD/SLICER/test_subject_1/real.mrml'
        slicer.util.loadScene(testPath)
        
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
