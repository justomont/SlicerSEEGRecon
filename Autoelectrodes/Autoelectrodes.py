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
    fidNode.GetNthFiducialPosition(n,pos)
    return pos


def findContacts(fidNode,checked_bipolar):
    
    # Get the aseg map
    global asegVolumeNode, asegVoxelArray
    asegVolumeNode = slicer.mrmlScene.GetFirstNodeByName('aseg')
    asegVoxelArray = slicer.util.arrayFromVolume(asegVolumeNode)
    
    # All markup's names and positions in RAS coordinates
    markup_names = [fidNode.GetNthFiducialLabel(i) for i in range(fidNode.GetNumberOfFiducials())]
    markup_RAS = [NthFiducialPosition(fidNode,i) for i in range(fidNode.GetNumberOfFiducials())]
    
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
    if any([test.count(electrode) for electrode in np.unique(test)]) == 1:
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
    end_markups, end_RAS = [list(tuple) for tuple in  end_tuples]
    
    #############################################################################
    # Monopolar. The tool just adds the contacts where they really are in space.#
    #############################################################################
    # Goal: Compute the position of the remaining contacts and include them in the monopolar markup list (real-R/L)
    
    # Initialize lists for the markups and their RAS coordinates
    global monopolar_markups, monopolar_RAS
    monopolar_markups = []
    monopolar_RAS = []
    
    monopolar_markups_WM = []
    monopolar_RAS_WM = []
    fidNodeWM = slicer.vtkMRMLMarkupsFiducialNode()
    fidNodeWM.SetName(fidNode.GetName()+"-WM")
    slicer.mrmlScene.AddNode(fidNodeWM)
    
    monopolar_markups_P = []
    monopolar_RAS_P = []
    fidNodeP = slicer.vtkMRMLMarkupsFiducialNode()
    fidNodeP.SetName(fidNode.GetName()+"-P")
    slicer.mrmlScene.AddNode(fidNodeP)
    
    monopolar_markups_E = []
    monopolar_RAS_E = []
    fidNodeE = slicer.vtkMRMLMarkupsFiducialNode()
    fidNodeE.SetName(fidNode.GetName()+"-ends")
    slicer.mrmlScene.AddNode(fidNodeE)
    
    # Iterate over all the markups defined by the user
    ruler_done_names = []
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
                    ijk_position = RAStoIJK(RAS[index], asegVolumeNode)
                    anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
                    if ("White" in anatomic_position) or ("WM-hypointensities" in anatomic_position):
                        monopolar_markups_WM.append(markup)
                        monopolar_RAS_WM.append(RAS[index])
                        fidNodeWM.AddFiducialFromArray(RAS[index], letter+str(digit))
                    else:
                        monopolar_markups_P.append(markup)
                        monopolar_RAS_P.append(RAS[index])
                        fidNodeP.AddFiducialFromArray(RAS[index], letter+str(digit))
                # calculate how many markups should be added until the next user-defined markup
                additions = next_digit - digit-1
                # calculate how distant the markups should be
                distance = np.subtract(RAS[index+1], RAS[index])/(additions+1)
                # add these new markups
                for i in range(additions):
                    new_digit = digit+i+1
                    # new_position = np.add(RAS[index], distance*(digit+i))
                    new_position = np.add(RAS[index], distance*(i+1))
                    monopolar_markups.append(letter+str(new_digit))
                    monopolar_RAS.append(new_position)
                    
                    fidNode.AddFiducialFromArray(new_position, letter+str(new_digit))
                    
                    ijk_position = RAStoIJK(new_position, asegVolumeNode)
                    anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
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
                ijk_position = RAStoIJK(new_position, asegVolumeNode)
                anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
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
                
                rulerName = letter
                if rulerName not in ruler_done_names: 
                    ruler_done_names.append(rulerName)
                else:
                    # print("else")
                    i=0
                    while rulerName in ruler_done_names:
                        i +=1
                        rulerName = '%s_%i' % (letter, i)
                    ruler_done_names.append(rulerName)
                rulerNode.SetName(rulerName)
                rulerNode.Initialize(slicer.mrmlScene)
                rulerNode.SetPosition1(RAS[index])
                rulerNode.SetPosition2(RAS[index+1])
                if RAS[index][0] > 0:
                    rulerNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
                else:
                    rulerNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
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
                
                rulerName = letter
                if rulerName not in ruler_done_names: 
                    ruler_done_names.append(rulerName)
                else:
                    # print("else")
                    i=0
                    while rulerName in ruler_done_names:
                        i +=1
                        rulerName = '%s_%i' % (letter, i)
                    ruler_done_names.append(rulerName)
                rulerNode.SetName(rulerName)
                rulerNode.Initialize(slicer.mrmlScene)
                rulerNode.SetPosition1(RAS[index])
                
                rulerNode.SetPosition2(P)
                if P[0] > 0:
                    rulerNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
                else:
                    rulerNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
                rulerNode.SetDistanceAnnotationScale(0)
                rulerNode.GetDisplayNode().SetLineThickness(6)
                rulerNode.GetDisplayNode().SetMaxTicks(0)
                rulerNode.SetLocked(True)
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
            
            rulerName = letter
            if rulerName not in ruler_done_names: 
                ruler_done_names.append(rulerName)
            else:
                # print("else")
                i=0
                while rulerName in ruler_done_names:
                    i +=1
                    rulerName = '%s_%i' % (letter, i)
                ruler_done_names.append(rulerName)
            rulerNode.SetName(rulerName)
            rulerNode.Initialize(slicer.mrmlScene)
            rulerNode.SetPosition1(RAS[index])
            
            rulerNode.SetPosition2(P)
            if P[0] > 0:
                rulerNode.GetDisplayNode().SetColor([0/255,85/255,225/255])
            else:
                rulerNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
            rulerNode.SetDistanceAnnotationScale(0)
            rulerNode.GetDisplayNode().SetLineThickness(6)
            rulerNode.GetDisplayNode().SetMaxTicks(0)
            rulerNode.SetLocked(True)
            
    # Lock all markups
    for markupN in range(fidNode.GetNumberOfMarkups()):
        fidNode.SetNthFiducialLocked(markupN,True)
        fidNode.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNodeWM.GetNumberOfMarkups()):
        fidNodeWM.SetNthFiducialLocked(markupN,True)
        fidNodeWM.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNodeP.GetNumberOfMarkups()):
        fidNodeP.SetNthFiducialLocked(markupN,True)
        fidNodeP.SetNthControlPointSelected(markupN,1)
    for markupN in range(fidNodeE.GetNumberOfMarkups()):
        fidNodeE.SetNthFiducialLocked(markupN,True)
        fidNodeE.SetNthControlPointSelected(markupN,1)
    
    if P[0] > 0:
        # SELECTED Color for the right hemisphere
        fidNode.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNodeWM.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNodeP.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        fidNodeE.GetDisplayNode().SetSelectedColor([0/255,85/255,225/255])
        # UNselected Color for the right hemisphere
        fidNode.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNodeWM.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNodeP.GetDisplayNode().SetColor([0/255,0/255,225/255])
        fidNodeE.GetDisplayNode().SetColor([0/255,0/255,225/255])
    else:
        fidNode.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNodeWM.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNodeP.GetDisplayNode().SetColor([170/255,0/255,0/255])
        fidNodeE.GetDisplayNode().SetColor([170/255,0/255,0/255])
    
    # Gliph type
    fidNode.GetDisplayNode().SetGlyphType(8)
    fidNode.GetDisplayNode().SetGlyphScale(5)
    fidNode.GetDisplayNode().SetTextScale(0)
    
    fidNodeWM.GetDisplayNode().SetGlyphType(8)
    fidNodeWM.GetDisplayNode().SetGlyphScale(3)
    fidNodeWM.GetDisplayNode().SetTextScale(0)
    fidNodeWM.GetDisplayNode().SetVisibility(False)
    
    fidNodeP.GetDisplayNode().SetGlyphType(8)
    fidNodeP.GetDisplayNode().SetGlyphScale(3)
    fidNodeP.GetDisplayNode().SetTextScale(0)
    fidNodeWM.GetDisplayNode().SetVisibility(False)
    
    fidNodeE.GetDisplayNode().SetGlyphType(8)
    fidNodeE.GetDisplayNode().SetGlyphScale(3)
    fidNodeE.GetDisplayNode().SetTextScale(0)
    fidNodeWM.GetDisplayNode().SetVisibility(False)
    
    fidNode.GetDisplayNode().SetVisibility(True)
    
    logging.info("Monopolar contact placement complete.\n") 
   
    
    if checked_bipolar:
        # Bipolar 
        fidNodeBi = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi.SetName("Bipolar-"+fidNode.GetName())
        slicer.mrmlScene.AddNode(fidNodeBi)
        bipolar_markups = []
        bipolar_RAS = []
        
        fidNodeBi_WM = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi_WM.SetName("Bipolar-"+fidNode.GetName()+"-WM")
        slicer.mrmlScene.AddNode(fidNodeBi_WM)
        bipolar_markups_WM = []
        bipolar_RAS_WM = []
        
        fidNodeBi_P = slicer.vtkMRMLMarkupsFiducialNode()
        fidNodeBi_P.SetName("Bipolar-"+fidNode.GetName()+"-P")
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
                    bi_tag = "-".join([letter+str(digit),monopolar_markups[index+1]])
                    bipolar_markups.append(bi_tag)
                    bipolar_RAS.append(middle_point)
                    
                    fidNodeBi.AddFiducialFromArray(middle_point, bi_tag)
                    
                    ijk_position = RAStoIJK(middle_point, asegVolumeNode)
                    anatomic_position = anatomicREL(asegVoxelArray[ijk_position[2],ijk_position[1],ijk_position[0]])
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
            fidNodeBi.SetNthControlPointSelected(markupN,0)
        for markupN in range(fidNodeBi_WM.GetNumberOfMarkups()):
            fidNodeBi_WM.SetNthFiducialLocked(markupN,True)
            fidNodeBi_WM.SetNthControlPointSelected(markupN,0)
        for markupN in range(fidNodeBi_P.GetNumberOfMarkups()):
            fidNodeBi_P.SetNthFiducialLocked(markupN,True)
            fidNodeBi_P.SetNthControlPointSelected(markupN,0)
        
        if P[0] > 0:
            # SELECTED color for the right hemisphere
            fidNodeBi.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            fidNodeBi_WM.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            fidNodeBi_P.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            # UNselected
            fidNodeBi.GetDisplayNode().SetColor([0/255,0/255,0/255])
            fidNodeBi_WM.GetDisplayNode().SetColor([0/255,0/255,0/255])
            fidNodeBi_P.GetDisplayNode().SetColor([0/255,0/255,0/255])
            
        else:
            # Selected
            fidNodeBi.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            fidNodeBi_WM.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            fidNodeBi_P.GetDisplayNode().SetSelectedColor([255/255,255/255,225/255])
            # Unselected
            fidNodeBi.GetDisplayNode().SetColor([0/255,0/255,0/255])
            fidNodeBi_WM.GetDisplayNode().SetColor([0/255,0/255,0/255])
            fidNodeBi_P.GetDisplayNode().SetColor([0/255,0/255,0/255])
        
        # Gliph type
        fidNodeBi.GetDisplayNode().SetGlyphType(8)
        fidNodeBi.GetDisplayNode().SetGlyphScale(3)
        fidNodeBi.GetDisplayNode().SetTextScale(0)
        
        fidNodeBi_WM.GetDisplayNode().SetGlyphType(8)
        fidNodeBi_WM.GetDisplayNode().SetGlyphScale(3)
        fidNodeBi_WM.GetDisplayNode().SetTextScale(0)
        
        fidNodeBi_P.GetDisplayNode().SetGlyphType(8)
        fidNodeBi_P.GetDisplayNode().SetGlyphScale(3)
        fidNodeBi_P.GetDisplayNode().SetTextScale(0)
        
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
        self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_bipolar.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.inputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputListSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_transfer.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.DirectoryButton_subject.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.SceneName.connect("textChanged(QString)", self.updateParameterNodeFromGUI)
        
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
        self.ui.fixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("fixedVolume"))
        self.ui.checkBox_bipolar.checked = (self._parameterNode.GetParameter("Bipolar") == "true")
        self.ui.checkBox_transfer.checked = (self._parameterNode.GetParameter("Transfer") == "false")
        self.ui.inputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputListVolume"))
        self.ui.outputListSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputListVolume"))
        self.ui.DirectoryButton.directory = str(self._parameterNode.GetParameter("Directory"))
        self.ui.DirectoryButton_subject.directory = str(self._parameterNode.GetParameter("Directory_Subject"))
        self.ui.SceneName.text = str(self._parameterNode.GetParameter("SceneName"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.applyButton.toolTip = "Generate electrodes"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume node"
            self.ui.applyButton.enabled = False

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
        self._parameterNode.SetNodeReferenceID("fixedVolume", self.ui.fixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputListVolume", self.ui.inputListSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputListVolume", self.ui.outputListSelector.currentNodeID)
        self._parameterNode.SetParameter("Directory", str(self.ui.DirectoryButton.directory))
        self._parameterNode.SetParameter("Directory_Subject", str(self.ui.DirectoryButton_subject.directory))
        self._parameterNode.SetParameter("SceneName", str(self.ui.SceneName.text))
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
        sceneName = self.ui.SceneName.text
        # print(sceneName)
        self.logic.save_info(destinyDirectory,sceneName)
        
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
        markup_names = [inputList.GetNthFiducialLabel(i) for i in range(inputList.GetNumberOfFiducials())]
        markup_RAS = [NthFiducialPosition(inputList,i) for i in range(inputList.GetNumberOfFiducials())]
        markup_selected = [inputList.GetNthControlPointSelected(i) for i in range(inputList.GetNumberOfFiducials())]
        
        # Just the selected 
        selected_markup_names = list(compress(markup_names, markup_selected))
        selected_markup_RAS = list(compress(markup_RAS, markup_selected))
        
        for i, markup_name in enumerate(selected_markup_names):
            # First copy the electrodes in the new list
            outputList.AddFiducialFromArray(selected_markup_RAS[i], markup_name)
            outputList.SetNthFiducialLocked(i,True)
            outputList.SetNthControlPointSelected(i,0)
            # If transfer is selected, remove those electrodes from the original list
            if transfer:
                available_markup_names = [inputList.GetNthFiducialLabel(i) for i in range(inputList.GetNumberOfFiducials())]
                #  This loop is needed because each time that an electrode is removed from the markup list, it updates, so the index oof the remaining electrodes change
                for index, og_markup_name in enumerate(available_markup_names):
                    if markup_name == og_markup_name:
                        inputList.RemoveNthControlPoint(index)
    
    def save_info(self,destinationDirectory,sceneName):
        
        if slicer.util.saveScene(destinationDirectory+"/"+sceneName+".mrml"):
          logging.info("All files saved to: {0}".format(destinationDirectory))
          
          os.makedirs(destinationDirectory+"/res/", exist_ok=True)
          os.makedirs(destinationDirectory+"/note/", exist_ok=True)
          os.makedirs(destinationDirectory+"/edit/", exist_ok=True)
          
          # Save the view from the 3D view
          viewNodeID = "vtkMRMLViewNode1"
          import ScreenCapture
          cap = ScreenCapture.ScreenCaptureLogic()
          view = cap.viewFromNode(slicer.mrmlScene.GetNodeByID(viewNodeID))
          view.mrmlViewNode().SetBackgroundColor(1,1,1)
          view.mrmlViewNode().SetBackgroundColor2(1,1,1)
          view.mrmlViewNode().SetAxisLabelsVisible(False)
          view.mrmlViewNode().SetBoxVisible(False)
          view.resetFocalPoint()
          cap.captureImageFromView(view, destinationDirectory+"/"+sceneName+".png")
          
          # Save all the elements of the scene
          childIds = vtk.vtkIdList()
          shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
          shNode.GetItemChildren(shNode.GetSceneItemID(), childIds) 
          
          for itemIdIndex in range(childIds.GetNumberOfIds()):
              shItemId = childIds.GetId(itemIdIndex)
              # Write node to file (if storable)
              dataNode = shNode.GetItemDataNode(shItemId)
              if dataNode and dataNode.IsA("vtkMRMLStorableNode") and dataNode.GetStorageNode():
                  # storageNode = dataNode.GetStorageNode()
                  # filename = os.path.basename(storageNode.GetFileName())
                  if dataNode.IsA("vtkMRMLScalarVolumeNode") or dataNode.IsA("vtkMRMLLabelMapVolumeNode"):
                      filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".nrrd"
                      dataNode.GetStorageNode().SetFileName(filepath)
                      slicer.util.exportNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLModelNode"):
                      filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".vtk"
                      dataNode.GetStorageNode().SetFileName(filepath)
                      slicer.util.saveNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLMarkupsFiducialNode"):
                      filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".fcsv"
                      dataNode.GetStorageNode().SetFileName(filepath)
                      slicer.util.saveNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLAnnotationRulerNode"):
                      filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".acsv"
                      dataNode.GetStorageNode().SetFileName(filepath)
                      slicer.util.saveNode(dataNode, filepath)
              elif (dataNode and dataNode.IsA("vtkMRMLStorableNode") and not dataNode.GetStorageNode()):
                  dataNode.AddDefaultStorageNode()
                  if dataNode.IsA("vtkMRMLScalarVolumeNode") or dataNode.IsA("vtkMRMLLabelMapVolumeNode"):
                      filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".nrrd"
                      dataNode.GetStorageNode().SetFileName(filepath) 
                      slicer.util.exportNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLModelNode"):
                      filepath = destinationDirectory + "/res/" + dataNode.GetName() + ".vtk"
                      dataNode.GetStorageNode().SetFileName(filepath) 
                      slicer.util.saveNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLMarkupsFiducialNode"):
                      filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".fcsv"
                      dataNode.GetStorageNode().SetFileName(filepath) 
                      slicer.util.saveNode(dataNode, filepath)
                  if dataNode.IsA("vtkMRMLAnnotationRulerNode"):
                      filepath = destinationDirectory + "/note/" + dataNode.GetName() + ".acsv"
                      dataNode.GetStorageNode().SetFileName(filepath) 
                      slicer.util.saveNode(dataNode, filepath)     
                    # if dataNode.IsA("vtkMRMLLinearTransformNode"):
                    #     filepath = destinationDirectory + "/edit/" + dataNode.GetName() + ".h5"
                    #     dataNode.GetStorageNode().SetFileName(filepath) 
                    #     slicer.util.saveNode(dataNode, filepath)    
          
          slicer.util.saveScene(destinationDirectory+"/"+sceneName+".mrml")
            
        else:
          logging.error("Files saving failed")
        

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
