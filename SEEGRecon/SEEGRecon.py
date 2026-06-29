from scipy import signal
import json

import logging
import os
import zipfile
import vtk
import ctk, DICOMLib
import tempfile
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

from NiBabelModelIO import NiBabelModelIOLogic

import re
import numpy as np
import csv
from itertools import compress
from scipy.ndimage import binary_dilation
import shutil
import os
from os import listdir
from os.path import isfile,join

import time

import warnings

import qt
from qt import QProgressDialog

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

def anatomicREL_Brodmann(tag):

    region = ['unknown',
            '1 UNKNOWN',
            'BA 20',
            'BA 38',
            'BA 21',
            'BA 36',
            'BA 11',
            'BA 37',
            'BA 25',
            'BA 12',
            'BA 19',
            'BA 47',
            'BA 22',
            'BA 18',
            'BA 10',
            'BA 17',
            'BA 32',
            'BA 24',
            'BA 46',
            'BA 45',
            'BA 33',
            'BA 44',
            'BA 6',
            'BA 23',
            'BA 42',
            'BA 41',
            'BA 30',
            'BA 43',
            'BA 29',
            'BA 26',
            'BA 31',
            'BA 4',
            'BA 1',
            'BA 9',
            'BA 39',
            'BA 2',
            'BA 3',
            'BA 40',
            'BA 7',
            'BA 8',
            'BA 5',
            '41 UNKNOWN',
            '42 UNKNOWN',
            '43 UNKNOWN',
            '101 UNKNOWN',
            'BA 20',
            'BA 38',
            'BA 21',
            'BA 36',
            'BA 11',
            'BA 37',
            'BA 25',
            'BA 12',
            'BA 19',
            'BA 47',
            'BA 22',
            'BA 18',
            'BA 10',
            'BA 17',
            'BA 32',
            'BA 24',
            'BA 46',
            'BA 45',
            'BA 33',
            'BA 44',
            'BA 6',
            'BA 23',
            'BA 42',
            'BA 41',
            'BA 30',
            'BA 43',
            'BA 29',
            'BA 26',
            'BA 31',
            'BA 4',
            'BA 1',
            'BA 9',
            'BA 39',
            'BA 2',
            'BA 3',
            'BA 40',
            'BA 7',
            'BA 8',
            'BA 5',
            '141 UNKNOWN',
            '142 UNKNOWN',
            '143 UNKNOWN']
    indx = [0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143]
    
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

def findContacts(fidNode, labelmapNode):

    # Get the aseg map
    global asegVolumeNode, asegVoxelArray
    # In the new versions of the module the aseg volume is called brain_segmentation but aseg in the old ones
    asegVolumeNode = labelmapNode
    if not asegVolumeNode: 
        asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('brain_segmentation')
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
# This function does the registration of the MNI template to the space of the patient 

    global mniPath, linearTransformNode
    
    # ICBM152  
    # Paths
    mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')

    # MNI 152 linear symmetric
    templatePath = os.path.join(mniPath, 'icbm_avg_152_t1_tal_lin.nii') # moving volume
    movingmaskPath = os.path.join(mniPath, 'icbm_avg_152_t1_tal_lin_mask.nii') # moving volume mask

    # MNI 152 nonlinear asymmetric 2009c
    # templatePath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_asym_09c.nii') # moving volume
    # movingmaskPath = os.path.join(mniPath, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii') # moving volume mask
    
    # Set parameters
    movingVolumeNode = slicer.util.loadVolume(templatePath,properties={"name":"ICBM152_T1","center":True})
    
    linearTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    linearTransformNode.SetName("Transform2MNI")
    outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolumeNode.SetName("ICBM152_registered")
    
    aseg = slicer.util.getFirstNodeByName("brain_segmentation")
    fixedmaskNode = slicer.mrmlScene.CopyNode(aseg)
    fixedmaskNode.SetName("aseg_mask")
    movingmaskNode = slicer.util.loadVolume(movingmaskPath,properties={"name":"MNI_mask","labelmap":True,"center":True})
    
    parameters = {}
    parameters["fixedVolume"] = fixedVolumeNode
    parameters["movingVolume"] = movingVolumeNode
    parameters["samplingPercentage"] = 0.02
    parameters["linearTransform"] = linearTransformNode
    parameters["outputVolume"] = outputVolumeNode
    parameters["initializeTransformMode"] = "useCenterOfHeadAlign"
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
    mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
    # path of the labelmap file of MNI regions
    labelmapPath = os.path.join(mniPath, 'mni_icbm152_CerebrA_tal_nlin_sym_09c.nii') # labelmap
    # path of the labelmap file of the Brodmann Areas
    brodmannPath = os.path.join(mniPath, 'Brodmann_Mai_Matajnik.nii') # labelmap

    # Select the transform from the MNI to patient registration 
    transform = slicer.util.getFirstNodeByName("Transform2MNI")
    
    # Transform the labelmap to match the patient volume
    labelmapVolumeNode = slicer.util.loadVolume(labelmapPath,properties={"name":"MNI_labels","labelmap":True,"center":True})
    labelmapVolumeNode.SetName("transformed_MNI_labels")
    # labelmapVolumeNode.ApplyTransformMatrix(transform.GetMatrixTransformToParent())
    labelmapVolumeNode.SetAndObserveTransformNodeID(transform.GetID())
    time.sleep(15)
    slicer.util.forceRenderAllViews()
    time.sleep(5)

    # Transform the Brodmann Areas labelmap to match the patient volume
    brodmannVolumeNode = slicer.util.loadVolume(brodmannPath,properties={"name":"Brodmann_labels","labelmap":True,"center":True})
    brodmannVolumeNode.SetName("transformed_Brodmann_Areas")
    brodmannVolumeNode.SetAndObserveTransformNodeID(transform.GetID())
    time.sleep(15)
    slicer.util.forceRenderAllViews()
    time.sleep(5)

    
def regionsMNI_2(destinyDirectory, markups_to_map, templateVolumeNode):  
    mniPath = os.path.join(os.path.dirname(__file__), 'Resources/MNI')
    # Select the transform from the MNI to patient registration 
    transform = slicer.util.getFirstNodeByName("Transform2MNI")
    labelmapVolumeNode = slicer.util.getFirstNodeByName("transformed_MNI_labels")
    brodmannVolumeNode = slicer.util.getFirstNodeByName("transformed_Brodmann_Areas")

    # Obtain voxel array of the label map to obtain the number associated to a specific location
    MNIVoxelArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    BrodmannVoxelArray = slicer.util.arrayFromVolume(brodmannVolumeNode)
    
    # Load MNI table relating number tag to area
    MNI_details = pd.read_csv(os.path.join(mniPath, 'CerebrA_LabelDetails.csv'))
    
    # Inverse transform to compute the MNI coordinates of each contact
    worldToMniTransform = vtk.vtkGeneralTransform()
    transform.GetTransformFromWorld(worldToMniTransform)

    if templateVolumeNode:
        templateVoxelArray = slicer.util.arrayFromVolume(templateVolumeNode)

    # Get the number of control points (number of contacts)
    n_contacts = markups_to_map.GetNumberOfControlPoints()

    # Initialize dataframe for the atlases
    if templateVolumeNode:
        atlas = pd.DataFrame(columns=['Contact', 'ASEG/Custom','Brodmann','MNI','X_mni', 'Y_mni', 'Z_mni'])
    else:
        atlas = pd.DataFrame(columns=['Contact','Brodmann','MNI','X_mni', 'Y_mni', 'Z_mni'])

    # Fill dataframe
    for i in range(n_contacts):

        contact_label = markups_to_map.GetNthControlPointLabel(i)
        ras = NthFiducialPosition(markups_to_map,i)
        # print(contact_label, ras)

        mni = [0,0,0]
        worldToMniTransform.TransformPoint(ras, mni)
        
        # Obtain Aseg Labels
        if templateVolumeNode:
            point_ijk = RAStoIJK(ras,templateVolumeNode)
            aseg_label = anatomicREL(templateVoxelArray[point_ijk[2],point_ijk[1],point_ijk[0]])

        # Obtain MNI Labels
        point_ijk = RAStoIJK(ras,labelmapVolumeNode)
        try:
            mni_label_number = MNIVoxelArray[point_ijk[2],point_ijk[1],point_ijk[0]]
        except:
            mni_label_number = 0
        
        # # In case that the label is non-existent. check surroundings
        # surround_index = 1
        # while mni_label_number == 0 and surround_index<5:
        #     surroundings = [-surround_index,0,surround_index]
        #     areas = []
        #     for x in surroundings:
        #         for y in surroundings:
        #             for z in surroundings:
        #                 try:
        #                     areas.append(MNIVoxelArray[point_ijk[2]+x,point_ijk[1]+y,point_ijk[0]+z])
        #                 except:
        #                     pass
        #     mni_label_number = max(set(areas), key = areas.count)
        #     surround_index = surround_index+1
        
        if mni_label_number != 0:
            mni_label = MNI_details[MNI_details.eq(mni_label_number).any(axis="columns")]["Label Name"].iloc[0]
        else:
            mni_label = "unknown"

        # Obtain Brodmann Area
        point_ijk = RAStoIJK(ras,brodmannVolumeNode)
        try:
            brodmann_area = anatomicREL_Brodmann(BrodmannVoxelArray[point_ijk[2],point_ijk[1],point_ijk[0]])
        except:
            brodmann_area = 'unknown'
        
        # Fill dataframe
        if templateVolumeNode:
            df = pd.DataFrame([[contact_label, aseg_label, brodmann_area, mni_label, mni[0], mni[1], mni[2]]], columns=['Contact', 'ASEG/Custom', 'Brodmann','MNI', 'X_mni', 'Y_mni', 'Z_mni'])
        else:
            df = pd.DataFrame([[contact_label, brodmann_area, mni_label, round(mni[0]), round(mni[1]), round(mni[2])]], columns=['Contact', 'Brodmann', 'MNI', 'X_mni', 'Y_mni', 'Z_mni'])
        atlas = pd.concat([atlas, df])

    # Save the files
    if destinyDirectory != '.':
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            atlas.to_csv(path_or_buf=destinyDirectory+"/location_electrodes.csv", index=False, index_label=False)
    
    print_atlas = atlas.to_string(index=False)
    print("\nLocation of electrodes:")
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
        print('UID',patientUID)
        for studyUID in studies:
            series = db.seriesForStudy(studyUID)
            print('study', studyUID)
            for seriesUID in series:
                DICOMLib.loadSeriesByUID([seriesUID])
        
#
# SEEGRecon
#

class SEEGRecon(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SEEGRecon"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Electrophysiology"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Justo Montoya-Gálvez (Pompeu Fabra University)","Alessandro Principe (Pompeu Fabra University, Hospital del Mar Research Institute)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/justomont/SlicerSEEGRecon">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Justo Montoya-Gálvez and Alessandro Principe and was partially funded by Agència de Gestió d’Ajuts Universitaris i de Recerca (AGAUR) Generalitat de Catalunya grant number FI-SDUR 20203.
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

    # SEEGRecon1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='SEEGRecon',
        sampleName='SEEGRecon1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'SEEGRecon1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='SEEGRecon1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='SEEGRecon1'
    )

    # SEEGRecon2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='SEEGRecon',
        sampleName='SEEGRecon2',
        thumbnailFileName=os.path.join(iconsPath, 'SEEGRecon2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='SEEGRecon2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='SEEGRecon2'
    )

#
# SEEGReconWidget
#

class SEEGReconWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = SEEGReconLogic()
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # slicer.util.selectModule("Volumes")

        # Load UI
        self.uiWidget = slicer.util.loadUI(self.resourcePath('UI/SEEGRecon.ui'))
        self.uiWidget.setMRMLScene(slicer.mrmlScene)

        # Always add to *this* module's layout
        self.layout.addWidget(self.uiWidget)

        # Keep ui reference
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)

        self.logic = SEEGReconLogic()
        self.initializeParameterNode()
        
        # Add observers for scene events
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.inputSelector
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.labelmapSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.labelmapSelector
        self.ui.labelmapSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.electrodesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.electrodesSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.electrodesSelector
        self.ui.electrodesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.fixedVolumeSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.fixedVolumeSelector
        self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.markupsToMapSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.markupsToMapSelector
        self.ui.markupsToMapSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputListSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.inputListSelector
        self.ui.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputListSelector = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.outputListSelector
        self.ui.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_transfer = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.transferCollapsibleButton.checkBox_transfer
        self.ui.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.DirectoryButton
        self.ui.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton_subject = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.DirectoryButton_subject
        self.ui.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_saveBundle.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton_saveBundle = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.DirectoryButton_saveBundle
        self.ui.DirectoryButton_saveBundle.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        # self.ui.DirectoryButton_rawFolder.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton_rawFolder = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.importTab.ImportGroupBox.DirectoryButton_rawFolder
        self.ui.DirectoryButton_rawFolder.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)

        self.ui.inputSelector_CT = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.registerTab.registerGroupBox.inputSelector_CT
        self.ui.inputSelector_CT.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_MRI = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.registerTab.registerGroupBox.inputSelector_MRI
        self.ui.inputSelector_MRI.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Other inputs
        # self.ui.saveFileName.editingFinished.connect(self.updateParameterNodeFromGUI)
        self.ui.saveFileName = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.saveFileName
        self.ui.saveFileName.connect('editingFinished()', self.updateParameterNodeFromGUI)

        # Visualization advanced inputs
        # self.ui.templateName.activated.connect(self.updateParameterNodeFromGUI)
        self.ui.templateName = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.visualizationCollapsibleButton.templateName
        self.ui.templateName.connect('activated(QString)', self.updateParameterNodeFromGUI)
        # self.ui.applySettingsButton.connect('clicked(bool)', self.onApplyTemplateSettingsButton)
        self.ui.applySettingsButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.visualizationCollapsibleButton.applySettingsButton
        self.ui.applySettingsButton.connect('clicked(bool)', self.onApplyTemplateSettingsButton)

        # Buttons
        self.ui.applyButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.inputsCollapsibleButton.applyButton
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # self.ui.pushButton.connect('clicked(bool)', self.onPushButton)
        self.ui.pushButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.pushButton
        self.ui.pushButton.connect('clicked(bool)', self.onPushButton)
        # self.ui.copyButton.connect('clicked(bool)', self.onCopyButton)
        self.ui.copyButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.copytransferCollapsibleButton.copyButton
        self.ui.copyButton.connect('clicked(bool)', self.onCopyButton)
        # self.ui.pushButton_mapping.connect('clicked(bool)', self.onPushButton_mapping)
        self.ui.pushButton_mapping = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton.pushButton_mapping
        self.ui.pushButton_mapping.connect('clicked(bool)', self.onPushButton_mapping)
        # self.ui.saveTableButton.connect('clicked(bool)', self.onSaveTableButton)
        self.ui.saveTableButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.mainTab.outputsCollapsibleButton_2.saveTableButton
        self.ui.saveTableButton.connect('clicked(bool)', self.onSaveTableButton)
        # self.ui.saveBundleButton.connect('clicked(bool)', self.onSaveBundleButton)
        self.ui.saveBundleButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.saveBundleButton
        self.ui.saveBundleButton.connect('clicked(bool)', self.onSaveBundleButton)

        self.ui.comboBoxSaveBundle = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.exportTab.exportGroupBox.comboBoxSaveBundle
        self.ui.comboBoxSaveBundle.connect('activated(QString)', self.updateParameterNodeFromGUI)

        self.ui.ImportPatientButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.importTab.ImportGroupBox.ImportPatientButton
        self.ui.ImportPatientButton.connect('clicked(bool)', self.onImportPatientButton)

        self.ui.ImportEpiDBButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.importTab.ImportEpiDBGroupBox.ImportEpiDBButton
        self.ui.ImportEpiDBButton.connect('clicked(bool)', self.onImportEpiDBButton)

        self.ui.registerButton = self.uiWidget.TabWidget.qt_tabwidget_stackedwidget.registerTab.registerGroupBox.registerButton
        self.ui.registerButton.connect('clicked(bool)', self.onRegisterButton)

        # Add the option to create a custom labelmap
        combo = self.ui.templateName
        CREATE_ITEM_TEXT = "+ Create new template..."
        # Add only once
        if combo.findText(CREATE_ITEM_TEXT) == -1:
            combo.addItem(CREATE_ITEM_TEXT)
        # Connect handler
        combo.connect('activated(QString)', self.onTemplateNameActivated)

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
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.labelmapSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputLabelMap"))
        self.ui.electrodesSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputElectrodes"))
        self.ui.fixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("fixedVolume"))
        self.ui.markupsToMapSelector.setCurrentNode(self._parameterNode.GetNodeReference("markupsToMap"))
        self.ui.checkBox_transfer.checked = (self._parameterNode.GetParameter("Transfer") == "false")
        self.ui.inputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputListVolume"))
        self.ui.outputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputListVolume"))
        self.ui.DirectoryButton.directory = str(self._parameterNode.GetParameter("Directory"))
        self.ui.DirectoryButton_subject.directory = str(self._parameterNode.GetParameter("Directory_Subject"))
        self.ui.DirectoryButton_saveBundle.directory = str(self._parameterNode.GetParameter("saveDirectory"))
        self.ui.DirectoryButton_rawFolder.directory = str(self._parameterNode.GetParameter("rawDirectory"))
        self.ui.EpiDB_filePath.currentPath = str(self._parameterNode.GetParameter("EpiDB_filePath"))
        self.ui.saveFileName.text = str(self._parameterNode.GetParameter("saveFileName"))
                
        # self.ui.visulizeMarkupsWidget.setCurrentNode(self._parameterNode.GetNodeReference("visualizationInputVolume"))
        self.ui.templateName.setCurrentText(str(self._parameterNode.GetParameter("templateName")))
        self.ui.comboBoxSaveBundle.setCurrentText(str(self._parameterNode.GetParameter("comboBoxSaveBundle")))

        self.ui.inputSelector_CT.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_CT"))
        self.ui.inputSelector_MRI.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_MRI"))

        # Update buttons states and tooltips (Enable)
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.applyButton.toolTip = "Generate electrodes"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume node"
            self.ui.applyButton.enabled = False

        if self._parameterNode.GetNodeReference("templateName"):
            self.ui.templateName.toolTip = "Apply template"
            self.ui.applySettingsButton.enabled = True
        else:
            self.ui.templateName.toolTip = "Select template"
            self.ui.applySettingsButton.enabled = False

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputElectrodes"):
            self.ui.saveTableButton.toolTip = "Generate table"
            self.ui.saveTableButton.enabled = True
        else:
            self.ui.saveTableButton.toolTip = "Select input"
            self.ui.saveTableButton.enabled = False

        if self._parameterNode.GetParameter("saveDirectory"):
            self.ui.saveBundleButton.enabled = True
        else:
            self.ui.saveBundleButton.enabled = False

        if self._parameterNode.GetParameter("rawDirectory"):
            self.ui.ImportPatientButton.enabled = True
        else:
            self.ui.ImportPatientButton.enabled = False

        if self._parameterNode.GetParameter("EpiDB_filePath"):
            self.ui.ImportEpiDBButton.enabled = True
        else:
            self.ui.ImportEpiDBButton.enabled = False

        if self._parameterNode.GetNodeReference("InputVolume_CT") and self._parameterNode.GetNodeReference("InputVolume_MRI"):
            self.ui.registerButton.toolTip = "Register CT Image to MR Image"
            self.ui.registerButton.enabled = True
        else:
            self.ui.registerButton.toolTip = "Select input volumes (CT and MRI)"
            self.ui.registerButton.enabled = False

        
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

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputLabelMap", self.ui.labelmapSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputElectrodes", self.ui.electrodesSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("fixedVolume", self.ui.fixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("markupsToMap", self.ui.markupsToMapSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputListVolume", self.ui.inputListSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputListVolume", self.ui.outputListSelector.currentNodeID)
        self._parameterNode.SetParameter("Directory", str(self.ui.DirectoryButton.directory))
        self._parameterNode.SetParameter("Directory_Subject", str(self.ui.DirectoryButton_subject.directory))
        self._parameterNode.SetParameter("saveDirectory", str(self.ui.DirectoryButton_saveBundle.directory))
        self._parameterNode.SetParameter("rawDirectory", str(self.ui.DirectoryButton_rawFolder.directory))
        self._parameterNode.SetParameter("EpiDB_filePath", str(self.ui.EpiDB_filePath.currentPath))
        self._parameterNode.SetParameter("Transfer", "true" if self.ui.checkBox_transfer.checked else "false")
        self._parameterNode.SetParameter("saveFileName", str(self.ui.saveFileName.text))

        self._parameterNode.SetNodeReferenceID("InputVolume_CT", self.ui.inputSelector_CT.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_MRI", self.ui.inputSelector_MRI.currentNodeID)

        # self._parameterNode.SetNodeReferenceID("visualizationInputVolume", self.ui.visulizeMarkupsWidget.currentNode().currentNodeID)
        self._parameterNode.SetParameter("templateName", str(self.ui.templateName.currentText))
        self._parameterNode.SetParameter("comboBoxSaveBundle", str(self.ui.comboBoxSaveBundle.currentText))
        

        self._parameterNode.EndModify(wasModified)

    def captureTemplateFromFirstVisible(self):
        markups = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        for markup in markups:
            displayNode = markup.GetDisplayNode()
            if displayNode and displayNode.GetVisibility() == 1:
                return {
                    "glyphSize": displayNode.GetGlyphScale(),
                    "textSize": displayNode.GetTextScale(),
                    "selectedColor": displayNode.GetSelectedColor(),
                    "unselectedColor": displayNode.GetColor(),
                    "activeColor": displayNode.GetActiveColor(),
                    "glyphType": displayNode.GetGlyphType(),
                }
        return None

    def onTemplateNameActivated(self, text):
        CREATE_ITEM_TEXT = "+ Create new template..."
        if text == CREATE_ITEM_TEXT:
            newLabel = qt.QInputDialog.getText(
                slicer.util.mainWindow(),
                "Create new label",
                "Enter template name:"
            )
            if newLabel:  # empty string if cancelled
                insertIndex = self.ui.templateName.findText(CREATE_ITEM_TEXT)
                self.ui.templateName.insertItem(insertIndex, newLabel)
                self.ui.templateName.setCurrentText(newLabel)
                settings = self.captureTemplateFromFirstVisible()
                if settings:
                    self.logic.userTemplates[newLabel] = settings
            else:
                # revert selection if cancelled
                self.ui.templateName.setCurrentIndex(0)
        self.updateParameterNodeFromGUI()

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fidNode = self.ui.inputSelector.currentNode()
            labelmapNode = self.ui.labelmapSelector.currentNode()

            print(labelmapNode)
            
            self.logic.process(fidNode, labelmapNode)

    def onRegisterButton(self):
        """
        Run processing when user clicks "Register" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            ctVolumeNode = self.ui.inputSelector_CT.currentNode()
            t1VolumeNode = self.ui.inputSelector_MRI.currentNode()

            self.logic.registerCTtoT1MRI(t1VolumeNode, ctVolumeNode)
    
    def onPushButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fixedVolumeNode = self.ui.fixedVolumeSelector.currentNode()
            self.logic.regions(fixedVolumeNode)
            
            destinyDirectory = self.ui.DirectoryButton.directory
            self.logic.regions_parttwo(destinyDirectory)
    
    def onApplyTemplateSettingsButton(self):
        
        # electrodeList = self.ui.visulizeMarkupsWidget.currentNode()

        templateName = self.ui.templateName.currentText

        self.logic.applyTemplate(templateName)
    
    def onPushButton_mapping(self):

        destinyDirectory = self.ui.DirectoryButton.directory
        markups_to_map = self.ui.markupsToMapSelector.currentNode()
        templateVolumeNode = self.ui.customTemplateSelector.currentNode()
        self.logic.regions_partthree(destinyDirectory, markups_to_map, templateVolumeNode)
        
            
    def onSaveTableButton(self):

        destinyDirectory = self.ui.DirectoryButton_subject.directory
        electrodesNode = self.ui.electrodesSelector.currentNode()

        self.logic.save_info(destinyDirectory, electrodesNode)
        
    def onSaveBundleButton(self):

        destinyDirectory = self.ui.DirectoryButton_saveBundle.directory
        name_of_file = self.ui.saveFileName.text
        format_Option = self.ui.comboBoxSaveBundle.currentText

        self.logic.save_Bundle(destinyDirectory, name_of_file, format_Option)
    
    def onImportPatientButton(self):

        rawDirectory = self.ui.DirectoryButton_rawFolder.directory
        print(f"Importing patient data from: {rawDirectory}")

        self.logic.import_patient(rawDirectory)

    def onImportEpiDBButton(self):

        EpiDB_filePath = self.ui.EpiDB_filePath.currentPath
        self.logic.import_EpiDB(EpiDB_filePath)

    def onCopyButton(self):

        inputList = self.ui.inputListSelector.currentNode()
        outputList = self.ui.outputListSelector.currentNode()
        checked_transfer = self.ui.checkBox_transfer.checked

        self.logic.copy_transfer(inputList, outputList, checked_transfer)
        
#
# SEEGReconLogic
#

class SEEGReconLogic(ScriptedLoadableModuleLogic):
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
        self.userTemplates = {}

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Bipolar"):
            parameterNode.SetParameter("Bipolar", "false")

    def process(self, fidNode, labelmapNode):
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

        findContacts(fidNode, labelmapNode)

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

            displayNode = markupNode.GetMarkupsDisplayNode()
            if not displayNode:
                markupNode.CreateDefaultDisplayNodes()
                displayNode = markupNode.GetMarkupsDisplayNode()

            displayNode.SetGlyphTypeFromString("Sphere3D")
            displayNode.SetSelectedColor(color)
            displayNode.SetColor(color)
            displayNode.SetUseGlyphScale(False) # disable screen-relative scaling
            displayNode.SetGlyphSize(5.0) # This sets the markups to 5 mm

            # Lock the markup node
            markupNode.SetLocked(True)

            # --- Reorder control points using natural sort on label ---
            controlPoints = []
            for i in range(numPoints):
                pos = [0.0, 0.0, 0.0]
                markupNode.GetNthControlPointPositionWorld(i, pos)
                label = markupNode.GetNthControlPointLabel(i)
                controlPoints.append((label, pos))
                markupNode.SetNthControlPointLocked(i, True)

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

        if templateName in self.userTemplates:
            self.applyUserTemplate(templateName)

    def applyUserTemplate(self, templateName):
        s = self.userTemplates.get(templateName)
        if not s:
            return
        markups = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        for markup in markups:
            displayNode = markup.GetDisplayNode()
            # Apply only to visible fiducials
            if displayNode.GetVisibility() != 1:
                continue
            if displayNode and displayNode.GetVisibility() == 1:
                displayNode.SetGlyphScale(s["glyphSize"])
                displayNode.SetTextScale(s["textSize"])
                displayNode.SetSelectedColor(s["selectedColor"])
                displayNode.SetColor(s["unselectedColor"])
                displayNode.SetActiveColor(s["activeColor"])
                displayNode.SetGlyphType(s["glyphType"])

    
    def regions(self,fixedVolumeNode):
        
        logging.info("Mapping electrodes into the MNI space...")
        registerMNI(fixedVolumeNode)
    
    def regions_parttwo(self,destinyDirectory):
        
        # time.sleep(15)
        regionsMNI(destinyDirectory)
        
    def regions_partthree(self,destinyDirectory, markups_to_map, templateVolumeNode):
        regionsMNI_2(destinyDirectory, markups_to_map, templateVolumeNode)
        
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
        print(asegVolumeNode.GetName())
        # if not asegVolumeNode: 
        #     asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('aseg')
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

    def registerCTtoT1MRI(self, t1VolumeNode, ctVolumeNode, progress=None, progress_lenght=None):

        """        Register CT volume to T1 MRI volume.
        :param t1VolumeNode: T1 MRI volume node
        :param ctVolumeNode: CT volume node
        
        Following: Registration of brain CT images to an MRI template for the purpose of lesion-symptom mapping, Hugo J. Kuijf 
        https://link.springer.com/chapter/10.1007/978-3-319-02126-3_12#citeas
        """

        volumesLogic = slicer.modules.volumes.logic()
        transformLogic = slicer.modules.transforms.logic()

        print("     Extracting bone from CT volume...")
        if progress:
            progress.setValue(progress_lenght+1)
            progress.setLabelText(f"This process could take up to 5 minutes. \nExtracting bone from CT volume...")
            slicer.app.processEvents()

        # Extract the bone from the CT volume
        ctVolume = volumesLogic.CloneVolume(slicer.mrmlScene,ctVolumeNode, 'CT_Bones')
        volumesLogic.CenterVolume(ctVolume)
        boneThreshold = 500
        ctArray = slicer.util.arrayFromVolume(ctVolume)
        ctArray[ctArray <= boneThreshold] = 0  # Set values below threshold to 0
        slicer.util.arrayFromVolumeModified(ctVolume)

        print("     Loading template...")
        if progress:
            progress.setValue(progress_lenght+2)
            progress.setLabelText(f"This process could take up to 5 minutes. \nLoading template...")
            slicer.app.processEvents()
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
        if progress:
            progress.setValue(progress_lenght+3)
            progress.setLabelText(f"This process could take up to 5 minutes. \nRegistering template to CT... This may take a while.")
            slicer.app.processEvents()
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
        if progress:
            progress.setValue(progress_lenght+4)
            progress.setLabelText(f"This process could take up to 5 minutes. \nRegistering Template to T1 MRI... This may take a while.")
            slicer.app.processEvents()

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
        if progress:
            progress.setValue(progress_lenght+5)
            progress.setLabelText(f"This process could take up to 5 minutes. \nApplying non-linear transforms to CT volume...")
            slicer.app.processEvents()
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
        if progress:
            progress.setValue(progress_lenght+6)
            progress.setLabelText(f"This process could take up to 5 minutes. \nRegistering Goal to CT... Almost finished!")
            slicer.app.processEvents()
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

        slicer.util.selectModule("AutoSEEG")

        # Delete all nodes that are not needed aymore
        slicer.mrmlScene.RemoveNode(ctVolume)
        slicer.mrmlScene.RemoveNode(mniVolume)
        slicer.mrmlScene.RemoveNode(mniBrainMaskVolume)
        slicer.mrmlScene.RemoveNode(mniHeadMaskVolume)
        slicer.mrmlScene.RemoveNode(mniBonesVolume)
        outputVolume = slicer.util.getNode("CT_postElectrodes")
        slicer.mrmlScene.RemoveNode(outputVolume)
        outputVolume = slicer.util.getNode("CTGoaltoCT")
        slicer.mrmlScene.RemoveNode(outputVolume)
        outputVolume = slicer.util.getNode("ICBM_CTregistered")
        slicer.mrmlScene.RemoveNode(outputVolume)
        outputVolume = slicer.util.getNode("ICBM_MRIregistered")
        slicer.mrmlScene.RemoveNode(outputVolume)
        slicer.mrmlScene.RemoveNode(ICBMtoCT_Transform)
        slicer.mrmlScene.RemoveNode(ICBMtoMRI_Transform)
        slicer.mrmlScene.RemoveNode(CTtoICBM_Transform)
        slicer.mrmlScene.RemoveNode(ctVolumeNode)
        ctRegisteredVolumeNode = slicer.util.getNode("CT_Registered_ICBM")
        slicer.mrmlScene.RemoveNode(ctRegisteredVolumeNode)
        slicer.mrmlScene.RemoveNode(ctGoalVolumeNode)
        slicer.mrmlScene.RemoveNode(ctRawVolumeNode)
        slicer.mrmlScene.RemoveNode(CTGoaltoCT_Transform)

        # Remove the patient that was loaded when loading the DICOM
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        sceneItemID = shNode.GetSceneItemID()
        childIDs = vtk.vtkIdList()
        shNode.GetItemChildren(sceneItemID, childIDs, True)
        for i in range(childIDs.GetNumberOfIds()):
            itemID = childIDs.GetId(i)
            if shNode.GetItemLevel(itemID) == "Patient":
                shNode.RemoveItem(itemID)

        print("     Done!")

    def import_EpiDB(self, filePath):

        print(filePath)
        # create temp dir
        tempDir = tempfile.mkdtemp()

        # Unzip bundle
        with zipfile.ZipFile(filePath, 'r') as z:
            z.extractall(tempDir)

        for root, _, files in os.walk(tempDir):
            for file in files:
                print(root, file)

                if file.endswith('.nrrd'):
                    if file.startswith('brain_segmentation'):
                        success, labelNode = slicer.util.loadVolume(os.path.join(root, file), {'labelmap': True,'center':True}, returnNode=True)
                        colorNode = slicer.util.getNode("FreeSurferLabels")
                        if not colorNode:
                            colorNode = slicer.util.getNode("GenericColors")
                        labelNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
                    else:
                        slicer.util.loadVolume(os.path.join(root, file), {'center':True})
                elif file.endswith('.vtk'):
                    slicer.util.loadModel(os.path.join(root, file))
                elif file.endswith(".mrk.json"):
                    slicer.util.loadMarkups(os.path.join(root, file))



    def import_patient(self, rawDirectory):
        

        print(rawDirectory)

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
        
        progress = qt.QProgressDialog(
            "This process could take up to 5 minutes.",
            "OK",
            0,
            len(found_files)+3+7,
            slicer.util.mainWindow()
        )
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(qt.Qt.ApplicationModal)
        progress.show()

        print('Loading DICOM data... \n')
        
        # Load DICOM data
        importAndLoadDICOMFolder(dicom_folder)
        progress.setValue(0)
        slicer.app.processEvents()

        # IMPORTANT: Since the DICOM is the only vtkMRMLScalarVolumeNode we look for it and change the name (it cannot be changed in the importAndLoadDICOMFolder function)
        # This should be always the first volume loaded, so it is safe to do this
        dicomVolumeNode = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")[0]
        if dicomVolumeNode:
            dicomVolumeNode.SetName("CT_postElectrodes_RAW")

            volumesLogic = slicer.modules.volumes.logic()
            volumesLogic.CenterVolume(dicomVolumeNode)

        for fileIndex, filename in enumerate(found_files):

            print(f"Processing file: {filename}")
            progress.setValue(fileIndex+1)
            progress.setLabelText(f"This process could take up to 5 minutes. \nProcessing file: {filename}...")
            slicer.app.processEvents()

            file_path = found_files[filename]

            # Load the segmentation file
            if filename.endswith('.mgz') and 'aseg' in filename.lower():
                print(f"Loading segmentation: {filename} ...\n")
                
                # Load aseg as labelmap volume
                labelmapNode = slicer.util.loadVolume(file_path, {'labelmap': True, 'center':True})
                labelmapNode.SetName("aseg")

                # Set color table to FreeSurfer labels
                fsColorNode = slicer.util.getNode('FreeSurferLabels')
                labelmapNode.GetDisplayNode().SetAndObserveColorNodeID(fsColorNode.GetID())

                # # Optional: show the segmentation
                # segmentationNode.CreateDefaultDisplayNodes()
                # segmentationNode.GetDisplayNode().SetVisibility2DFill(True)

            if filename.endswith('.mgz') and ('norm' in filename.lower() or 'T1' in filename):
                print(f"Loading volume: {filename} ...\n")
                volumeNode = slicer.util.loadVolume(file_path, {'center':True})
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

        self.registerCTtoT1MRI(t1VolumeNode, ctVolumeNode, progress, progress_lenght=len(found_files)+2)
        
        progress.setValue(len(found_files)+3+7)
        slicer.app.processEvents()
        

        print("Files imported successfully!\n")


        # Count time it took to import the patient data
        stopTime = time.time()
        elapsed_minutes = int((stopTime - startTime) // 60)
        elapsed_seconds = int((stopTime - startTime) % 60)
        print(f'Importing patient data completed in {elapsed_minutes} min {elapsed_seconds} sec')

        progress.close()




    def save_Bundle(self, destinationDirectory, name_of_file, format_Option):
        
        # Create folder of the patient
        destinationDirectory = os.path.join(destinationDirectory, name_of_file)

        namesOfFilesToSearch = ['aseg',
                                'brain_segmentation',
                                '3D',
                                'MRI_T1',
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
                                'PET',
                                'ct.3d',
                                'CT.3d',
                                'ct.3D',
                                'CT.3D',
                                'ctp.3d',
                                'CTp.3d',
                                'ctp.3D',
                                'CTp.3D',
                                'CT_postElectrodes',
                                'res',
                                'resection',
                                'real-L','real-R',
                                'real-L-P','real-R-P',
                                'real-L_Grey_Matter', 'real-R_Grey_Matter',
                                # 'real-L-W','real-R-W',
                                # 'real-L_White_Matter', 'real-R_White_Matter',
                                'electrodes_L','electrodes_R',
                                'electrodes_L_Grey_Matter','electrodes_R_Grey_Matter',
                                'electrodes_L_grey','electrodes_R_grey',
                                # 'electrodes_L_White_Matter','electrodes_R_White_Matter',
                                # 'real-R_outside_brain', 'real-L_outside_brain',
                                'electrodes_bipolar_L', 'electrodes_bipolar_R',
                                'electrodes_bipolar_L_grey','electrodes_bipolar_R_grey',
                                'Bipolar-real-L','Bipolar-real-R',
                                'Bipolar_real-L', 'Bipolar_real-R',
                                'Bipolar-real-L-P','Bipolar-real-R-P',
                                'Bipolar_real-L_Grey_Matter','Bipolar_real-R_Grey_Matter',
                                # 'Bipolar_real-L_White_Matter','Bipolar_real-R_White_Matter',
                                'Bipolar_electrodes_L','Bipolar_electrodes_R',
                                'Bipolar_electrodes_L_Grey_Matter','Bipolar_electrodes_R_Grey_Matter',
                                # 'Bipolar_electrodes_L_White_Matter','Bipolar_electrodes_R_White_Matter',
                                # 'Bipolar_real-L_Outside','Bipolar_real-R_Outside',
                                'etc','ETC']
        
        nrrdFiles = ['aseg', 'brain_segmentation','3D','MRI_T1', 'brain', 'norm',
                     'pet.3d','pet.3D','PET.3D','PET.3d','PET',
                     'ct.3d','CT.3d','ct.3D','CT.3D',
                     'ctp.3d','CTp.3d','ctp.3D','CTp.3D','CT_postElectrodes']
        
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
                        'brain_segmentation':'brain_segmentation',
                        '3D': 'MRI_T1',
                        'MRI_T1':'MRI_T1',
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
                        'PET':'PET',
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
                        'CT_postElectrodes':'CT_postElectrodes',
                        'res':'resection',
                        'resection':'resection',
                        'real-L':'electrodes_L',
                        'real-R':'electrodes_R',
                        'real-L-P':'electrodes_L_grey',
                        'real-R-P':'electrodes_R_grey',
                        'real-L_Grey_Matter':'electrodes_L_grey',
                        'real-R_Grey_Matter':'electrodes_R_grey',
                        'real-L-W': 'electrodes_L_white',
                        'real-R-W': 'electrodes_R_white',
                        'real-L_White_Matter': 'electrodes_L_white', 
                        'real-R_White_Matter': 'electrodes_R_white',
                        'electrodes_L':'electrodes_L',
                        'electrodes_R':'electrodes_R',
                        'electrodes_L_grey':'electrodes_L_grey',
                        'electrodes_R_grey':'electrodes_R_grey',
                        'electrodes_L_Grey_Matter':'electrodes_L_grey',
                        'electrodes_R_Grey_Matter':'electrodes_R_grey',
                        'electrodes_L_White_Matter': 'electrodes_L_white',
                        'electrodes_R_White_Matter': 'electrodes_R_white',
                        'real-R_outside_brain': 'electrodes_R_outside', 
                        'real-L_outside_brain': 'electrodes_L_outside',
                        'electrodes_bipolar_L':'electrodes_bipolar_L',
                        'electrodes_bipolar_R':'electrodes_bipolar_R',
                        'Bipolar-real-L':'electrodes_bipolar_L',
                        'Bipolar-real-R':'electrodes_bipolar_R',
                        'Bipolar_real-L':'electrodes_bipolar_L',
                        'Bipolar_real-R':'electrodes_bipolar_R',
                        'electrodes_bipolar_L_grey':'electrodes_bipolar_L_grey',
                        'electrodes_bipolar_R_grey':'electrodes_bipolar_R_grey',
                        'Bipolar-real-L-P':'electrodes_bipolar_L_grey',
                        'Bipolar-real-R-P':'electrodes_bipolar_R_grey',
                        'Bipolar_electrodes_L':'electrodes_bipolar_L',
                        'Bipolar_electrodes_R':'electrodes_bipolar_R',
                        'Bipolar_electrodes_L_Grey_Matter':'electrodes_bipolar_L_grey',
                        'Bipolar_electrodes_R_Grey_Matter':'electrodes_bipolar_R_grey',
                        'Bipolar_electrodes_L_White_Matter': 'electrodes_bipolar_L_white',
                        'Bipolar_electrodes_R_White_Matter': 'electrodes_bipolar_R_white',
                        'Bipolar_real-L_Grey_Matter':'electrodes_bipolar_L_grey',
                        'Bipolar_real-R_Grey_Matter':'electrodes_bipolar_R_grey',
                        'Bipolar_real-L_White_Matter': 'electrodes_bipolar_L_white',
                        'Bipolar_real-R_White_Matter': 'electrodes_bipolar_R_white',
                        'Bipolar_real-L_Outside': 'electrodes_bipolar_L_outside',
                        'Bipolar_real-R_Outside': 'electrodes_bipolar_R_outside',
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

        if format_Option in ('Compressed', 'Both'):
            print(format_Option)
            # Path of the resulting ZIP file (same parent directory)
            zip_path = destinationDirectory + ".epiDB"
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(destinationDirectory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Preserve relative folder structure inside ZIP
                        arcname = os.path.relpath(file_path, destinationDirectory)
                        zipf.write(file_path, arcname)
            if format_Option == 'Compressed':
                # Delete folders
                shutil.rmtree(destinationDirectory)


            

#
# SEEGReconTest
#

class SEEGReconTest(ScriptedLoadableModuleTest):
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
        self.test_SEEGRecon1()

    def test_SEEGRecon1(self):
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
        # inputVolume = SampleData.downloadSample('SEEGRecon1')
        
        testPath = '/Volumes/GoogleDrive/Mi unidad/_PhD/SLICER/test_subject_1/real.mrml'
        slicer.util.loadScene(testPath)
        
        self.delayDisplay('Loaded test data set')

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # Test the module logic

        logic = SEEGReconLogic()

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
