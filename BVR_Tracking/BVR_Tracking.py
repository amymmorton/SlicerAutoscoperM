import os
from typing import Optional

import numpy as np
import slicer
import vtk
from slicer import vtkMRMLModelNode, vtkMRMLSequenceNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

#
# BVR_Tracking
#


class BVR_Tracking(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BVR_Tracking")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Tracking"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Amy Morton "]
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#BVR_Tracking">module documentation</a>.
"""
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""
        )

        # Additional initialization step after application startup is complete

    # slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # BVR_Tracking1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BVR_Tracking",
        sampleName="BVR_Tracking1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "BVR_Tracking1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="BVR_Tracking1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="BVR_Tracking1",
    )

    # BVR_Tracking2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="BVR_Tracking",
        sampleName="BVR_Tracking2",
        thumbnailFileName=os.path.join(iconsPath, "BVR_Tracking2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="BVR_Tracking2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="BVR_Tracking2",
    )


#
# BVR_TrackingParameterNode
#


@parameterNodeWrapper
class BVR_TrackingParameterNode:
    """
    The parameters needed by module.

    modelList - model list loaded from file
    traList - tracking file list loaded from file
    selectedModel - model from modelList for tra assignment
    selectedTraSeq - traSeq from traList for assignment to selectedModel
    """

    modelList: list[str] = []  # List of model names to be used in the module
    traList: list[str] = []  # List of tracking file names to be used in the module
    selectedModel: vtkMRMLModelNode
    selectedTraSeq: vtkMRMLSequenceNode


#
# BVR_TrackingWidget
#


class BVR_TrackingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BVR_Tracking.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BVR_TrackingLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        # Load
        self.ui.loadModels_pb.connect("clicked(bool)", self.onloadModels_pb)
        self.ui.loadTras_pb.connect("clicked(bool)", self.onloadTras_pb)
        # Apply transforms

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

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
        if not self._parameterNode.selectedModel:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstModelNode:
                self._parameterNode.selectedModel = firstModelNode

        if not self._parameterNode.selectedTraSeq:
            firstSeqNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceNode")
            if firstSeqNode:
                self._parameterNode.selectedTraSeq = firstSeqNode

    def setParameterNode(self, inputParameterNode: Optional[BVR_TrackingParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.selectedModel and self._parameterNode.selectedTraSeq:
            self.ui.loadTras_pb.toolTip = _("Assign tra to model")
            self.ui.loadTras_pb.enabled = True
        else:
            self.ui.loadTras_pb.toolTip = _("Load/Select model and tra files ")
            self.ui.loadTras_pb.enabled = False

    def onloadModels_pb(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Load stls
            slicer.util.openAddModelDialog()
            self.ui.inputModelSelector.setCurrentNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode"))

    def onloadTras_pb(self) -> None:
        traFileDirectory = (
            self.ui.trackingFileSelector.currentPath if self.ui.trackingFileSelector.currentPath else None
        )
        model = self.ui.inputModelSelector.currentNode() if self.ui.inputModelSelector.currentNode() else None
        self.logic.loadTras(traFileDirectory, model)


#
# BVR_TrackingLogic
#


class BVR_TrackingLogic(ScriptedLoadableModuleLogic):
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
        return BVR_TrackingParameterNode(super().getParameterNode())

    def loadTras(
        self,
        traFile,
        model,
    ) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param traFile: this tra file to be loaded as Seq
        """
        # new sequence browser if one does not exist
        seqBrowser = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceBrowserNode")
        if not seqBrowser:
            seqBrowser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "Tracking_Browser")

        tName = os.path.splitext(os.path.basename(traFile))[0]
        tra = np.loadtxt(traFile, delimiter=",")
        tra.resize(tra.shape[0], 4, 4)

        tf = self.initializeTransforms(tra, tName, seqBrowser)
        # print(seqNode.GetNumberOfDataNodes())

        model.SetAndObserveTransformNodeID(tf.GetID())

    def initializeTransforms(self, tform4x4, tName, seqBrowser) -> slicer.vtkMRMLLinearTransformNode:
        """Creates a new transform sequence in the scene browser ."""

        newSequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", f"{tName}_transform_sequence")
        seqBrowser.AddSynchronizedSequenceNode(newSequenceNode)

        identityTfm = slicer.mrmlScene.CreateNodeByClass("vtkMRMLLinearTransformNode")
        identityTfm.UnRegister(None)  # release extra reference of object to avoid memory leak message

        # batch the processing event for the addition of the new transform nodes, for speedup
        slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)

        for i in range(tform4x4.shape[0]):
            tform_i = slicer.util.vtkMatrixFromArray(tform4x4[i, :, :])
            identityTfm.SetMatrixTransformToParent(tform_i)
            newSequenceNode.SetDataNodeAtValue(identityTfm, str(i))

        slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
        slicer.app.processEvents()
        return identityTfm

    def loadTraAsVTK(data: np.ndarray) -> list[vtk.vtkMatrix4x4]:
        """
        Converts the tracking data to a list of vtkMatrix4x4.

        :param data: The tracking data.

        :return: The tracking data as a sequence.
        """
        _, cols = data.shape

        EXPECTED_DIMENSION = 16
        if cols != EXPECTED_DIMENSION:
            # Check to see if the data was exported as a 4x4 matrix, probably want to expand this method
            # to support other formats.
            slicer.util.errorDisplay("Loading as sequence currently only supports 4x4 matrices")
            return None

        result = []
        for idx, row in enumerate(data):
            matrix = vtk.vtkMatrix4x4()
            # If there is no data, set the matrix to the previous matrix.
            # If its the first matrix, set it to the identity matrix.
            if np.isnan(row).any():
                if idx == 0:
                    matrix.Identity()
                else:
                    matrix.DeepCopy(result[idx - 1])
            else:
                for i in range(4):
                    for j in range(4):
                        matrix.SetElement(i, j, row[i * 4 + j])
            result.append(matrix)
        return result


#
# BVR_TrackingTest
#


class BVR_TrackingTest(ScriptedLoadableModuleTest):
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
        self.test_BVR_Tracking1()

    def test_BVR_Tracking1(self):
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

        self.delayDisplay("Test passed")
