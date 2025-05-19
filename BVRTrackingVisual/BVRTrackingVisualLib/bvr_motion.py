import glob
import os

import numpy as np
import slicer
import vtk

from TrackingEvaluationLib import data as dt


class ModelInMotion (ModelData):
    """
    Class to store model sets and corresponding motion data.
    
    :param modelNode: stl model (AUT space)
    :param userSequenceData: AUT tra tracked dat as np array: 
    """

def __init__(self, modelNode: str, userSequenceData: np.ndarray):
        self.userModelNode = modelNode
        self.userTransformNode = None
        self.userDisplayNode = self.userModelNode.GetDisplayNode()
        self.userSequence = loadTraAsSequence(userSequenceData)

        self.setColor()  # Set the initial color to white(default value)

def initializeTransforms(self):
    self.userTransformNode = slicer.vtkMRMLTransformNode()
    slicer.mrmlScene.AddNode(self.userTransformNode)
    self.userModelNode.SetAndObserveTransformNodeID(self.userTransformNode.GetID())

def cleanup(self):
    """
    Clean up the model data.
    """
    slicer.mrmlScene.RemoveNode(self.userModelNode)
    slicer.mrmlScene.RemoveNode(self.userTransformNode)

def updateTransform(self, index: int):
    """
    Update the transform of the model node.

    :param index: The index of the transform.
    """
    if self.userSequence is None:
        slicer.util.errorDisplay("No user tracking data!")
        return
    if index >= len(self.userSequence):
        slicer.util.errorDisplay("Index out of bounds!")
        return
    self.userTransformNode.SetMatrixTransformToParent(self.userSequence[index])
    

def setColor(self, rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """
    Update the color of the model node.

    :param rgb: The RGB color to set.
    """
    self.userDisplayNode.SetColor(rgb[0], rgb[1], rgb[2])

class BVR3DScene(Scene):
    """
    Class to store the scene data.

    :param models: List of model n
    :param userSequenceFileName: The filename of the user sequence.
    """
         