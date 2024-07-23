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
from os.path import isfile,join

import time

import warnings

import qt

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

def findContacts(fidNode,checked_bipolar):
    
    # Get the aseg map
    global asegVolumeNode, asegVoxelArray
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
    right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rhp')
    right_hemisphere_Volume = computeBrainVolume(right_hemisphere_VolumeNode)
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
    
    # If the user specified that they want the bipolar representation
    if checked_bipolar: 
        
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
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.electrodesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_bipolar.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        
        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.pushButton.connect('clicked(bool)', self.onPushButton)
        self.ui.copyButton.connect('clicked(bool)', self.onCopyButton)
        self.ui.pushButton_mapping.connect('clicked(bool)', self.onPushButton_mapping)
        self.ui.saveButton.connect('clicked(bool)', self.onSaveButton)
        

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
        self.ui.electrodesSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.fixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("fixedVolume"))
        self.ui.checkBox_bipolar.checked = (self._parameterNode.GetParameter("Bipolar") == "true")
        self.ui.checkBox_transfer.checked = (self._parameterNode.GetParameter("Transfer") == "false")
        self.ui.inputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputListVolume"))
        self.ui.outputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputListVolume"))
        self.ui.DirectoryButton.directory = str(self._parameterNode.GetParameter("Directory"))
        self.ui.DirectoryButton_subject.directory = str(self._parameterNode.GetParameter("Directory_Subject"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.applyButton.toolTip = "Generate electrodes"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume node"
            self.ui.applyButton.enabled = False
            
        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.saveButton.toolTip = "Generate table"
            self.ui.saveButton.enabled = True
        else:
            self.ui.saveButton.toolTip = "Select input"
            self.ui.saveButton.enabled = False

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
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.electrodesSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("fixedVolume", self.ui.fixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputListVolume", self.ui.inputListSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputListVolume", self.ui.outputListSelector.currentNodeID)
        self._parameterNode.SetParameter("Directory", str(self.ui.DirectoryButton.directory))
        self._parameterNode.SetParameter("Directory_Subject", str(self.ui.DirectoryButton_subject.directory))
        self._parameterNode.SetParameter("Bipolar", "true" if self.ui.checkBox_bipolar.checked else "false")
        self._parameterNode.SetParameter("Transfer", "true" if self.ui.checkBox_transfer.checked else "false")

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fidNode = self.ui.inputSelector.currentNode()
            checked_bipolar = self.ui.checkBox_bipolar.checked
            
            # checked_mapping = self.ui.checkBox_mapping.checked
            fixedVolumeNode = self.ui.fixedVolumeSelector.currentNode()
            
            self.logic.process(fidNode,checked_bipolar)
    
    def onPushButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            fixedVolumeNode = self.ui.fixedVolumeSelector.currentNode()
            self.logic.regions(fixedVolumeNode)
            
            destinyDirectory = self.ui.DirectoryButton.directory
            self.logic.regions_parttwo(destinyDirectory)
    
    def onPushButton_mapping(self):
        
        destinyDirectory = self.ui.DirectoryButton.directory
        self.logic.regions_partthree(destinyDirectory)
        
            
    def onSaveButton(self):
        
        destinyDirectory = self.ui.DirectoryButton_subject.directory
        electrodesNode = self.ui.electrodesSelector.currentNode()        
        
        # print(sceneName)
        self.logic.save_info(destinyDirectory, electrodesNode)
        
    def onCopyButton(self):
        
        inputList = self.ui.inputListSelector.currentNode()
        outputList = self.ui.outputListSelector.currentNode()
        checked_transfer = self.ui.checkBox_transfer.checked
        
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

    def process(self, fidNode,checked_bipolar):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param fidNode: Markup node where the start and ending contacts of the electrodes to be processed are
        :param fixedVolumeNode: Volume node to be used as fixed volume for the MNI registration of the brain
        :param checked_bipolar: boolean to compute the bipolar montage of the electrodes too
        """

        if not fidNode:
            raise ValueError("Input volume or fixed volume is invalid")

        startTime = time.time()
        logging.info("Processing electrodes...")
        
        findContacts(fidNode,checked_bipolar)
        
        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    
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
        
        global asegVolumeNode, asegVoxelArray
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
        right_hemisphere_VolumeNode = slicer.mrmlScene.GetFirstNodeByName('rhp')
        right_hemisphere_Volume = computeBrainVolume(right_hemisphere_VolumeNode)
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
            
        
        # if slicer.util.saveScene(destinationDirectory+"/"+sceneName+".mrml"):
        #   logging.info("File saved to: {0}".format(destinationDirectory))
          
        #   os.makedirs(destinationDirectory+"/res/", exist_ok=True)
        #   os.makedirs(destinationDirectory+"/note/", exist_ok=True)
        #   os.makedirs(destinationDirectory+"/edit/", exist_ok=True)
          
        #   # Save the view from the 3D view
        #   viewNodeID = "vtkMRMLViewNode1"
        #   import ScreenCapture
        #   cap = ScreenCapture.ScreenCaptureLogic()
        #   view = cap.viewFromNode(slicer.mrmlScene.GetNodeByID(viewNodeID))
        #   view.mrmlViewNode().SetBackgroundColor(1,1,1)
        #   view.mrmlViewNode().SetBackgroundColor2(1,1,1)
        #   view.mrmlViewNode().SetAxisLabelsVisible(False)
        #   view.mrmlViewNode().SetBoxVisible(False)
        #   view.resetFocalPoint()
        #   cap.captureImageFromView(view, destinationDirectory+"/"+sceneName+".png")
          
        #   # Save all the elements of the scene
        #   childIds = vtk.vtkIdList()
        #   shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        #   shNode.GetItemChildren(shNode.GetSceneItemID(), childIds) 
          
        #   for itemIdIndex in range(childIds.GetNumberOfIds()):
        #       shItemId = childIds.GetId(itemIdIndex)
        #       # Write node to file (if storable)
        #       dataNode = shNode.GetItemDataNode(shItemId)
        #       if dataNode and dataNode.IsA("vtkMRMLStorableNode") and dataNode.GetStorageNode():
        #           # storageNode = dataNode.GetStorageNode()
        #           # filename = os.path.basename(storageNode.GetFileName())
        #           if dataNode.IsA("vtkMRMLScalarVolumeNode") or dataNode.IsA("vtkMRMLLabelMapVolumeNode"):
        #               filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".nrrd"
        #               dataNode.GetStorageNode().SetFileName(filepath)
        #               slicer.util.exportNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLModelNode"):
        #               filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".vtk"
        #               dataNode.GetStorageNode().SetFileName(filepath)
        #               slicer.util.saveNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLMarkupsFiducialNode"):
        #               filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".fcsv"
        #               dataNode.GetStorageNode().SetFileName(filepath)
        #               slicer.util.saveNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLAnnotationRulerNode"):
        #               filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".acsv"
        #               dataNode.GetStorageNode().SetFileName(filepath)
        #               slicer.util.saveNode(dataNode, filepath)
        #       elif (dataNode and dataNode.IsA("vtkMRMLStorableNode") and not dataNode.GetStorageNode()):
        #           dataNode.AddDefaultStorageNode()
        #           if dataNode.IsA("vtkMRMLScalarVolumeNode") or dataNode.IsA("vtkMRMLLabelMapVolumeNode"):
        #               filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".nrrd"
        #               dataNode.GetStorageNode().SetFileName(filepath) 
        #               slicer.util.exportNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLModelNode"):
        #               filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".vtk"
        #               dataNode.GetStorageNode().SetFileName(filepath) 
        #               slicer.util.saveNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLMarkupsFiducialNode"):
        #               filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".fcsv"
        #               dataNode.GetStorageNode().SetFileName(filepath) 
        #               slicer.util.saveNode(dataNode, filepath)
        #           if dataNode.IsA("vtkMRMLAnnotationRulerNode"):
        #               filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".acsv"
        #               dataNode.GetStorageNode().SetFileName(filepath) 
        #               slicer.util.saveNode(dataNode, filepath)     
        #             # if dataNode.IsA("vtkMRMLLinearTransformNode"):
        #             #     filepath = destinationDirectory + "/edit/" + dataNode.GetName() + ".h5"
        #             #     dataNode.GetStorageNode().SetFileName(filepath) 
        #             #     slicer.util.saveNode(dataNode, filepath)    
          
        #   slicer.util.saveScene(destinationDirectory+"/"+sceneName+".mrml")
            
        # else:
        #   logging.error("Files saving failed")
        

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
