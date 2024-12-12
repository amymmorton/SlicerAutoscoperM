import glob
import logging
import os
from typing import Annotated, Optional

import pathlib
import vtk
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLModelNode
from slicer import vtkMRMLMarkupsROINode


#
# roiFromModelBounds
#


class roiFromModelBounds(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("roiFromModelBounds")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Tracking"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Amy Morton (Brown University)"]  
        
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
WIP modeule part of SlicerAutoscoperM
Intended to act as foundation fro subvolume 3d-3d volume registration 
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
Template for this file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # roiFromModelBounds1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="roiFromModelBounds",
        sampleName="roiFromModelBounds1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "roiFromModelBounds1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="roiFromModelBounds1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="roiFromModelBounds1",
    )

    # roiFromModelBounds2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="roiFromModelBounds",
        sampleName="roiFromModelBounds2",
        thumbnailFileName=os.path.join(iconsPath, "roiFromModelBounds2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="roiFromModelBounds2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="roiFromModelBounds2",
    )


#
# roiFromModelBoundsParameterNode
#


@parameterNodeWrapper
class roiFromModelBoundsParameterNode:
    """
    The parameters needed by module.

    modelFile_path- load several stl inputModels from file

    inputVolume- Voume for ROI crop
    modelROI - The output volume that will contain the thresholded volume.
    croppedVolume - The output volume that will be .
    """
    modelFile_path: pathlib.Path
    inputVolume: vtkMRMLScalarVolumeNode
    modelBounds: float
    modelROI: vtkMRMLMarkupsROINode
    croppedVolume: vtkMRMLScalarVolumeNode


#
# roiFromModelBoundsWidget
#


class roiFromModelBoundsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/roiFromModelBounds.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = roiFromModelBoundsLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        #self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.showBounds_pb.connect("clicked(bool)", self.onShowModelBoundsButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        #if not self._parameterNode.inputModel:
        #    firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
        #    if firstModelNode:
        #        self._parameterNode.inputModel = firstModelNode

        #if not self._parameterNode.inputVolume:
        #    firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #    if firstVolumeNode:
        #        self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[roiFromModelBoundsParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)
            self._checkCanGenModelROI()

    def _checkCanGenModelROI(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.modelFile_path:
            self.ui.showBounds_pb.toolTip = _("Compute output volume")
            self.ui.showBounds_pb.enabled = True
        else:
            self.ui.showBounds_pb.toolTip = _("Select input ModelPath and volume nodes")
            self.ui.showBounds_pb.enabled = True

   
    def onShowModelBoundsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            volumeNode = self.ui.inputVolSelector.currentNode()
            modelFileDir = self.ui.modelPath_lineEdit.currentPath
            
            #TO DO write validatePaths logic funciton
            #    if not self.logic.validatePaths(modelFileDir=modelFileDir):
            #        raise ValueError("Invalid paths")
            #        return
            modelFiles = glob.glob(os.path.join(modelFileDir, "*.*"))
            
            for indx, file in enumerate(modelFiles):
                modelNode = slicer.util.loadNodeFromFile(file)
                modelNode.CreateDefaultDisplayNodes()
                self.logic.modelBounds(modelNode,volumeNode, 
                                    self.ui.modelROISelector, self.ui.cropVolSelector)
 
                #self.logic.modelBounds(self.ui.inputSelector.currentNode(),self.ui.inputVolumeSelector, 
                #                    self.ui.modelROISelector, self.ui.cropVolSelector)

#
# roiFromModelBoundsLogic
#


class roiFromModelBoundsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return roiFromModelBoundsParameterNode(super().getParameterNode())

    def modelBounds(self,
                inputModel:vtkMRMLModelNode,
                inputVolume: vtkMRMLScalarVolumeNode,
                modelROI: vtkMRMLMarkupsROINode,
                croppedVolume: vtkMRMLScalarVolumeNode) -> None:

        tB = [0.0,0.0,0.0,0.0,0.0,0.0]
        inputModel.GetBounds(tB)
        #print(tB)
        #modelC = [0.0]*3
        #modelSize = [0.0]*3
        #strange that the model nodes have a getBounds- but no center and size..
        #and the roi has bno.. set Bounds- just center and size

        #numpy for array operatots:
        tnp_min = np.array([tB[0],tB[2],tB[4]])
        tnp_max = np.array([tB[1],tB[3],tB[5]])

        tnp_C = (tnp_min+tnp_max)/2
        tnpS = (tnp_min-tnp_max)

        #inputModel.GetCenter(modelC)
        #inputModel.GetSize(modelSize)
        mname = inputModel.GetName()

        #use tB lims to size roi
        #Create ROI node
        modelROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode") 
        modelROI.SetCenter(tnp_C.tolist())
        modelROI.SetSize(tnpS.tolist())
        modelROI.CreateDefaultDisplayNodes()  # only needed for display

        roi_name =mname+"_roi"
        modelROI.SetName(roi_name)

        #populate in Model ROI Output..?
        
        #print(modelROI)




#
# roiFromModelBoundsTest
#


class roiFromModelBoundsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_roiFromModelBounds1()

    def test_roiFromModelBounds1(self):
        """Ideally you should have several levels of tests.  At the lowest level
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

        import SampleData

        registerSampleData()
        inputModel = SampleData.downloadSample("roiFromModelBounds1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputModel.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = roiFromModelBoundsLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputModel, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputModel, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
