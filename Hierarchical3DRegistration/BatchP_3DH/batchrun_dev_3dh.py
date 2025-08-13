import os
import glob
import time

import slicer
import DICOMLib
from DICOMLib import DICOMUtils
from DICOM import DICOMWidget #slicer.util.selectModule("DICOM")

from Hierarchical3DRegistration import Hierarchical3DRegistrationLogic

def run_3dh_registration(model_folder, m_hier_inorder, source_vol_n, seq_n, tform_fold):
    """
    Run the 3DH registration process.
    
    Parameters:
    model_folder (str): Path to the folder containing the models.
    source_vol_n (str): Path to the source volume file.
    seq_n (str): Path to the sequence file.
    tform_fold (str): Path to the folder where transforms will be saved.
    m_hier_inorder (list): List of model names in hierarchical order.

    output:
    - Transforms_3DH .tra
    """

    



# Run the registration
#run_3dh_registration(model_folder[0], m_hier_inorder, source_vol_n, seq_n, tform_fold)

def run_3dh_registration(model_folder, m_hier_inorder, source_vol_n, seq_n, tform_fold, tform_ot):
    """
    Run the 3DH registration process.
    
    Parameters:
    model_folder (str): Path to the folder containing the models.
    source_vol_n (str): Path to the source volume file.
    seq_n (str): Path to the sequence file.
    tform_fold (str): Path to the folder where transforms will be saved.
    m_hier_inorder (list): List of model names in hierarchical order.

    output:
    - Transforms_3DH .tra
    """
    slicer.mrmlScene.Clear(0)

    m_3dh = slicer.modules.hierarchical3dregistration
    mWidget = m_3dh.widgetRepresentation().self()

    #switch to the 3DH module
    hd = slicer.util.selectModule("Hierarchical3DRegistration")

    m_nodes =[]
    # Load models
    for model_name in m_hier_inorder:
        model_path = os.path.join(model_folder[0], model_name)
        if os.path.exists(model_path):
            m = slicer.util.loadModel(model_path)
            m_nodes.append(m)
        
        else:
            print(f"Model {model_name} not found in {model_folder}")

    #construct model hierarchy, assign to 
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    rad_node = slicer.util.getNode("rad")
    tpm_node = slicer.util.getNode("tpm")
    mc1_node = slicer.util.getNode("mc1")

    itemID = shNode.GetItemByDataNode(rad_node)
    shNode.SetItemParent(shNode.GetItemByDataNode(tpm_node), shNode.GetItemByDataNode(rad_node))
    shNode.SetItemParent(shNode.GetItemByDataNode(mc1_node),shNode.GetItemByDataNode(tpm_node))


    mWidget.ui.SubjectHierarchyComboBox.setCurrentItem(itemID)
    mWidget.ui.ioDir.setCurrentPath(tform_fold)

    #load in 
    # Load source volume
    svol_node = slicer.util.loadVolume(source_vol_n)
    # Set the source volume in the parameter node
    #mLogic = mWidget.logic
    #mNode = mLogic.getParameterNode()

    mWidget.logic.getParameterNode().sourceVolume =  svol_node


    # Load sequence
    seq_n = slicer.util.loadSequence(seq_n)
    # Set the sequence in the parameter node
    mWidget.logic.getParameterNode().volumeSequence = seq_n

    while not mWidget.logic.getParameterNode().volumeSequence:
        time.sleep(1)

    mWidget.logic.getParameterNode().skipManualTfmAdjustments = True

    #set the current and end frames- they show in the ui- but are not being 
    #set in to the parameter node

    mWidget.logic.getParameterNode().startFrameIdx = 1
    mWidget.logic.getParameterNode().endFrameIdx = mWidget.logic.getParameterNode().volumeSequence.GetNumberOfDataNodes() - 1
    mWidget.onInitializeButton()

    mWidget.onImportButton(mWidget.rootBone)
    mWidget.doNextRegistrationStep()


    #cha'nge for export
    mWidget.ui.ioDir.setCurrentPath(tform_ot)

    mWidget.onExportButton()



"""
For debugging in Slicer, use the following to manipulate the module objects:

mWidget = slicer.modules.meniscussignalintensity.widgetRepresentation().self()
mLogic = mWidget.logic
mNode = mLogic.getParameterNode()

"""
m_hier_inorder = ['rad.stl', 'tpm.stl', 'mc1.stl']  # Example model names


# collect folder names 
folder_names = glob.glob(r"P:\SlicerAutoscoper^M\Data\3DCT\SUBJECTS\*")

folders = []
for i in folder_names:
    if 'BN' in i:
        folders.append(i)

#for folder in folders:
folder = folders[0]
#subj name is folder.split('\\')[-1]
subj_name = folder.split('\\')[-1]
#seqence, source vol, hierarchy, transforms folder & tra
model_folder = os.path.join(folder, 'Models')
mnames = ['rad.stl','tpm.stl','mc1.stl']
source_vol_n = os.path.join(folder, 'Scene','1 neutral.nrrd')
seq_n = os.path.join(folder, 'Scene','3DH',subj_name +'.seq.mrb')

tform_fold= os.path.join(folder, 'Scene','3DH','Transforms')
tform_ot = os.path.join(folder,'Tracking','OBB')
#create if not exist
if not os.path.exists(tform_ot):
    os.makedirs(tform_ot)

# Run the registration
run_3dh_registration(model_folder, m_hier_inorder, source_vol_n, seq_n, tform_fold, tform_ot)