# Automatic Myocardial Scar Segmentation from Multi-Sequence Cardiac MRI using Fully Convolutional Densenet with Inception and Squeeze-Excitation Module
Automatic and accurate myocardial scar segmentation from
multiple-sequence cardiac MRI is essential for the diagnosis and prognosis of patients with myocardial infarction. However, this is difficult
due to motion artifact, low contrast between scar and blood pool in late
gadolinium enhancement (LGE) MRI, and poor contrast between edema
and healthy myocardium in T2 cardiac MRI. In this paper, we proposed
a fully-automatic scar segmentation method using a cascaded segmentation network of three Fully Convolutional Densenet (FC-Densenet) with
Inception and Squeeze-Excitation module. It is called Cascaded FCDISE.
The first FCDISE is used to extract the region of interest and the second FCDISE to segment myocardium and the last one to segment scar
from the pre-segmented myocardial region. In the proposed segmentation
network, the inception module is incorporated at the beginning of the
network to extract multi-scale features from the input image, whereas
the squeeze-excitation blocks are placed in the skip connections of the
network to transfer recalibrated feature maps from the encoder to the
decoder. To encourage higher order similarities between predicted image
and ground truth, we adopted a dual loss function composed of logarithmic Dice loss and region mutual information (RMI) loss. Our method is
evaluated on the Multi-sequence CMR based Myocardial Pathology Segmentation challenge (MyoPS 2020) dataset. On the test set, our fullyautomatic approach achieved an average Dice score of 0.565 for scar and
0.664 for scar+edema. This is higher than the inter-observer variation
of manual scar segmentation. The proposed method outperformed similar methods and showed that adding the two modules to FC-Densenet
improves the segmentation result with little computational overhead.
