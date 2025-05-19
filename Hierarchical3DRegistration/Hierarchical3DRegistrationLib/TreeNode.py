from __future__ import annotations

import logging
import os

import slicer
import vtk

from AutoscoperM import IO, AutoscoperMLogic


class TreeNode:
    """
    Data structure to store a basic tree hierarchy along with data needed for registration.
    Each TreeNode represents a bone in the hierarchy and keeps track of its respective
    moving and fixed images for registration as well as the registration result transforms.
    """

    def __init__(
        self,
        hierarchyID: int,
        ctSequence: slicer.vtkMRMLSequenceNode,
        sourceVolume: slicer.vtkMRMLScalarVolumeNode,
        parent: TreeNode | None = None,
        isRoot: bool = False,
    ):
        """
        Create a new TreeNode with the given parent.

        :param hierarchyID: this node's subject hierarchy ID
        :param ctSequence: the entire volume sequence (target) to be registered
        :param sourceVolume: the scalar volume to register from (from which this node's model was generated)
        :param parent: reference to the parent node
        :param isRoot: whether this node it the root of the hierarchy
        """
        self.hierarchyID = hierarchyID
        self.isRoot = isRoot
        self.parent = parent

        if self.parent is not None and self.isRoot:
            raise ValueError("Node cannot be root and have a parent")

        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.name = self.shNode.GetItemName(self.hierarchyID)
        self.model = self.shNode.GetItemDataNode(self.hierarchyID)

        if self.model.GetClassName() != "vtkMRMLModelNode":
            raise ValueError(f"Hierarchy item '{self.name}' is not of type vtkMRMLModelNode!")

        # disable model rendering, it will be turned visible once this bone is registered
        self.model.CreateDefaultDisplayNodes()
        self.model.SetDisplayVisibility(False)

        # initialize registration inputs and outputs for this node
        self.sourceVolume = sourceVolume
        self.roi = self._generateRoiFromModel()
        self.croppedSourceVolume = AutoscoperMLogic.cropVolumeFromROI(sourceVolume, self.roi)
        self.croppedSourceVolume.SetName(f"{sourceVolume.GetName()}_{self.name}_cropped_source")
        self.transformSequence = self._initializeTransforms(ctSequence)
        self.croppedCtSequence = self._initializeCroppedCT(ctSequence)

        # initialize temporary variable to save the recent transform before manual adjustment
        self.currTransformBeforeAdjustment = slicer.mrmlScene.CreateNodeByClass("vtkMRMLLinearTransformNode")
        self.currTransformBeforeAdjustment.UnRegister(None)  # release extra reference to avoid memory leak message

        # recursively create any child nodes from the subject hierarchy
        children_ids = []
        self.shNode.GetItemChildren(self.hierarchyID, children_ids)
        self.childNodes = [
            TreeNode(hierarchyID=child_id, parent=self, ctSequence=ctSequence, sourceVolume=sourceVolume)
            for child_id in children_ids
        ]

    def _generateRoiFromModel(
        self,
        inputModel: slicer.vtkMRMLModelNode = None,
        inputVolume: slicer.vtkMRMLScalarVolumeNode = None,
    ) -> slicer.vtkMRMLMarkupsROINode:
        """Creates a region of interest node from this TreeNode's model."""
        import SegmentStatistics
        import vtk

        if inputModel is None:
            inputModel = self.model
        if inputVolume is None:
            inputVolume = self.sourceVolume

        segNode = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segNode)
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(inputModel, segNode)
        # Compute centroids

        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatParams = segStatLogic.getParameterNode()
        segStatParams.SetParameter("Segmentation", segNode.GetID())
        segStatParams.SetParameter("LabelmapSegmentStatisticsPlugin.centroid_ras.enabled", str(True))
        segStatParams.SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_x.enabled", str(True))
        segStatParams.SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_y.enabled", str(True))
        segStatParams.SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_z.enabled", str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        sid = stats.get("SegmentIDs")
        centroid = stats[(sid[0], "LabelmapSegmentStatisticsPlugin.centroid_ras")]
        principal_axis_x = stats[(sid[0], "LabelmapSegmentStatisticsPlugin.principal_axis_x")]
        principal_axis_y = stats[(sid[0], "LabelmapSegmentStatisticsPlugin.principal_axis_y")]
        principal_axis_z = stats[(sid[0], "LabelmapSegmentStatisticsPlugin.principal_axis_z")]

        vT = vtk.vtkMatrix4x4()
        vT.Identity()

        # Set the first three rows of the matrix with the vector components
        for i in range(3):
            vT.SetElement(0, i, principal_axis_x[i])
            vT.SetElement(1, i, principal_axis_y[i])
            vT.SetElement(2, i, principal_axis_z[i])
            vT.SetElement(i, 3, centroid[i])

        # Create a transform node and set the transform to it
        tfm_pc = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        tfm_pc.SetMatrixTransformToParent(vT)

        # Clone the node
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        itemIDToClone = shNode.GetItemByDataNode(self.model)
        clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
        clonedNode = shNode.GetItemDataNode(clonedItemID)

        # apply to model clone
        clonedNode.SetAndObserveTransformNodeID(tfm_pc.GetID())
        clonedNode.HardenTransform()

        # Create ROI node
        bb_center, bb_size = AutoscoperMLogic.getModelBoundingBox(clonedNode)
        modelROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        modelROI.SetCenter(bb_center)
        modelROI.SetSize(bb_size)
        modelROI.SetDisplayVisibility(0)
        modelROI.SetName(f"{self.name}_roi")

        # inverse tfm
        tfm_pc.Inverse()
        modelROI.SetAndObserveTransformNodeID(tfm_pc.GetID())
        modelROI.HardenTransform()

        # remove the transform node, seg node and the cloned model node
        slicer.mrmlScene.RemoveNode(tfm_pc)  # remove the transform node
        slicer.mrmlScene.RemoveNode(clonedNode)  # remove the cloned model node
        slicer.mrmlScene.RemoveNode(segNode)  # remove the segmentation node

        return modelROI

    def _initializeTransforms(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new transform sequence in the same browser as the CT sequence."""

        newSequenceNode = AutoscoperMLogic.createSequenceNodeInBrowser(f"{self.name}_transform_sequence", ctSequence)
        identityTfm = slicer.mrmlScene.CreateNodeByClass("vtkMRMLLinearTransformNode")
        identityTfm.UnRegister(None)  # release extra reference of object to avoid memory leak message

        # batch the processing event for the addition of the new transform nodes, for speedup
        slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)

        for i in range(ctSequence.GetNumberOfDataNodes()):
            idxValue = ctSequence.GetNthIndexValue(i)
            newSequenceNode.SetDataNodeAtValue(identityTfm, idxValue)

        slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
        slicer.app.processEvents()
        return newSequenceNode

    def _initializeCroppedCT(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new (but empty) volume sequence in the same browser as the CT sequence."""
        newSequenceNode = AutoscoperMLogic.createSequenceNodeInBrowser(
            f"{self.name}_cropped_volume_sequence", ctSequence
        )
        emptyVolume = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
        emptyVolume.UnRegister(None)  # release extra reference of object to avoid memory leak message

        # batch the processing event for the addition of the new transform nodes, for speedup
        slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)

        for i in range(ctSequence.GetNumberOfDataNodes()):
            idxValue = ctSequence.GetNthIndexValue(i)
            newSequenceNode.SetDataNodeAtValue(emptyVolume, idxValue)

        slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
        slicer.app.processEvents()
        return newSequenceNode

    def startInteraction(self, frameIdx) -> None:
        """Enable model visibility and transform interaction for this bone in the current frame"""
        current_tfm = self.getTransform(frameIdx)
        self.currTransformBeforeAdjustment.CopyContent(current_tfm, True)

        self.roi.SetAndObserveTransformNodeID(current_tfm.GetID())
        self.model.SetAndObserveTransformNodeID(current_tfm.GetID())
        self.model.SetDisplayVisibility(True)
        model_display = self.model.GetDisplayNode()
        model_display.SetVisibility2D(True)
        model_display.SetVisibility3D(True)

        model_center, _ = AutoscoperMLogic.getModelBoundingBox(self.model)

        tfm = self.getTransform(frameIdx)
        tfm.SetCenterOfTransformation(model_center)
        tfm.CreateDefaultDisplayNodes()
        tfm_display = tfm.GetDisplayNode()
        tfm_display.SetEditorVisibility(True)
        tfm_display.SetEditorVisibility3D(True)

        slicer.app.processEvents()

    def stopInteraction(self, frameIdx) -> vtk.vtkMatrix4x4:
        """Disable transform interaction for this bone in the current frame"""
        self.model.SetDisplayVisibility(True)
        model_display = self.model.GetDisplayNode()
        model_display.SetVisibility2D(True)
        model_display.SetVisibility3D(True)

        tfm = self.getTransform(frameIdx)
        tfm.CreateDefaultDisplayNodes()
        tfm_display = tfm.GetDisplayNode()
        tfm_display.SetEditorVisibility(False)
        tfm_display.SetEditorVisibility3D(False)

        # calculate the component of the transform modified by the user interaction
        current_tfm = self.getTransform(frameIdx)
        current_tfm_matrix = vtk.vtkMatrix4x4()
        current_tfm.GetMatrixTransformToParent(current_tfm_matrix)

        inverse_tfm_before_adjustment_matrix = vtk.vtkMatrix4x4()
        self.currTransformBeforeAdjustment.GetMatrixTransformToParent(inverse_tfm_before_adjustment_matrix)
        inverse_tfm_before_adjustment_matrix.Invert()

        tfm_manual = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(current_tfm_matrix, inverse_tfm_before_adjustment_matrix, tfm_manual)

        # reset the helper variable with the identity for next time
        identity_matrix = vtk.vtkMatrix4x4()
        self.currTransformBeforeAdjustment.SetMatrixTransformToParent(identity_matrix)

        slicer.app.processEvents()

        return tfm_manual

    def setupFrame(self, frameIdx, ctFrame) -> None:
        """
        Crop the target CT frame based on the initial guess transform and this node's ROI.
        """

        current_tfm = self.getTransform(frameIdx)
        current_model = self.model
        current_roi = self.roi
        self.roi.SetAndObserveTransformNodeID(current_tfm.GetID())

        # next check if the roi bounds exceed the CT frame target volume
        cut_model = AutoscoperMLogic.checkROIAndVolumeOverlap(self.roi, self.model, ctFrame)
        if cut_model is not None:
            logging.info(f"Using downsized ROI for '{self.name}' to not exceed target volume '{ctFrame.GetName()}'")
            # apply the inverse of the transform from initial position so that
            # the cut model is aligned with source volume, not current frame
            current_tfm.Inverse()
            cut_model.SetAndObserveTransformNodeID(current_tfm.GetID())
            cut_model.HardenTransform()
            current_tfm.Inverse()
            # update roi to dimension of the intersection and align it
            current_model = cut_model
            current_roi = self._generateRoiFromModel(cut_model, ctFrame)

        # generate the cropped source volume based on the current frame's roi
        slicer.mrmlScene.RemoveNode(self.croppedSourceVolume)
        current_roi.SetAndObserveTransformNodeID(None)
        self.croppedSourceVolume = AutoscoperMLogic.cropVolumeFromROI(self.sourceVolume, current_roi)
        self.croppedSourceVolume.SetName(f"{self.sourceVolume.GetName()}_{self.name}_cropped_source")
        current_roi.SetAndObserveTransformNodeID(current_tfm.GetID())

        # generate cropped target volume from the given frame
        current_model.SetAndObserveTransformNodeID(current_tfm.GetID())
        self.croppedSourceVolume.SetAndObserveTransformNodeID(current_tfm.GetID())
        croppedFrameNode = self.getCroppedFrame(frameIdx)
        AutoscoperMLogic.cropVolumeFromROI(ctFrame, current_roi, croppedFrameNode)

        if cut_model is not None:
            #    # delete clone of model that was cut, as well as the ROI for it
            slicer.mrmlScene.RemoveNode(cut_model)
            slicer.mrmlScene.RemoveNode(current_roi)

    def getTransform(self, idx: int) -> slicer.vtkMRMLTransformNode:
        """Returns the transform at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]

    def getCroppedFrame(self, idx: int) -> slicer.vtkMRMLScalarVolumeNode:
        if idx >= self.croppedCtSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return AutoscoperMLogic.getItemInSequence(self.croppedCtSequence, idx)[0]

    def _applyTransform(self, idx: int, transform: slicer.vtkMRMLTransformNode) -> None:
        """Applies and hardends a transform node to the transform sequence at the provided index."""
        current_transform = self.getTransform(idx)
        if current_transform is None:
            return
        current_transform.SetAndObserveTransformNodeID(transform.GetID())
        current_transform.HardenTransform()

    def applyTransformToChildren(self, idx: int, transform: slicer.vtkMRMLLinearTransformNode) -> None:
        """Applies the transform at the provided index to all children of this node."""
        for childNode in self.childNodes:
            childNode._applyTransform(idx, transform)
            # recurse down all child nodes and apply it to them as well
            childNode.applyTransformToChildren(idx, transform)

    def copyTransformToNextFrame(self, currentIdx: int) -> None:
        """Copies the transform at the provided index to the next frame."""
        import vtk

        currentTransform = self.getTransform(currentIdx)
        transformMatrix = vtk.vtkMatrix4x4()
        currentTransform.GetMatrixTransformToParent(transformMatrix)

        nextIdx = currentIdx + 1
        nextTransform = self.getTransform(nextIdx)
        if nextTransform is not None:
            nextTransform.SetMatrixTransformToParent(transformMatrix)

    def setModelsVisibility(self, visibility: bool) -> None:
        """Sets the visibility of the model node of this node and all its child nodes"""
        self.model.SetDisplayVisibility(visibility)
        for childNode in self.childNodes:
            childNode.setModelsVisibility(visibility)

    def setTransformFromMatrix(self, transform: vtk.vtkMatrix4x4, idx: int) -> None:
        """Sets the registration transform for the given index from the input matrix"""
        current_transform = self.getTransform(idx)
        if current_transform is None:
            raise ValueError(f"Could not set transform at index {idx} of '{self.transformSequence.GetName()}'.")
        current_transform.SetMatrixTransformToParent(transform)
        self.model.SetAndObserveTransformNodeID(current_transform.GetID())

    def exportTransformsAsTRAFile(self, exportDir: str):
        """Exports the sequence as a TRA file for reading into Autoscoper."""
        # Convert the sequence to a list of vtkMatrices
        transforms = []
        for idx in range(self.transformSequence.GetNumberOfDataNodes()):
            mat = vtk.vtkMatrix4x4()
            node = self.getTransform(idx)
            node.GetMatrixTransformToParent(mat)
            transforms.append(mat)

        if not os.path.exists(exportDir):
            os.mkdir(exportDir)
        filename = os.path.join(exportDir, f"{self.name}-abs-RAS.tra")
        IO.writeTRA(filename, transforms)

    def importTransfromsFromTRAFile(self, filename: str):
        """Loads a TRA file as the registration transform sequence"""
        import numpy as np

        tra = np.loadtxt(filename, delimiter=",")
        tra.resize(tra.shape[0], 4, 4)
        for idx in range(tra.shape[0]):
            self.setTransformFromMatrix(slicer.util.vtkMatrixFromArray(tra[idx, :, :]), idx)
