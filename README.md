## High-order Multilayer Attention Fusion Network for 3D Object Detection [\[paper\]](https://XXXXX
This network propose a hight-order feature fusion module and a multi-level attetion fusion module. The former captures hight-level 
semantic features fromsummed element-wise features, and then extracts the most salient and average features of the high-order fuser features, 
which are re-fused with the hight-order features through a weighted attention machanism.


## Updates
2024-08-06: High-order Multilayer Attention Fusion Network for 3D Object Detection V1.1 is released!


```

# Introduction

Three-dimensional object detection based on the fusion of 2D image data and 3D point clouds has become a research hotspot in the field of 3D scene understanding. However, different sensor data have discrepancies in spatial position, scale, and alignment, which severely impact detection performance. Inappropriate fusion methods can lead to the loss and interference of valuable information. Therefore, we propose the High-Order Multi-Level Attention Fusion Network (HMAF-Net), which takes camera images and voxelized point clouds as inputs for 3D object detection. To enhance the expressive power between different modality features, we introduce a high-order feature fusion module that performs multi-level convolution operations on the element-wise summed features. By incorporating filtering and non-linear activation, we extract deep semantic information from the fused multi-modal features. To maximize the effectiveness of the fused salient feature information, we introduce an attention mecha-nism that dynamically evaluates the importance of pooled features at each level, enabling adaptive weighted fusion of significant and secondary features. To validate the effectiveness of HMAF-Net, we conduct experiments on the KITTI dataset. In the 'Car', 'Pedestrian', and 'Cyclist' categories, HMAF-Net achieves mAP performances of 81.78%, 60.09%, and 63.91%, respectively, demonstrating more stable performance compared to other multi-modal methods. Furthermore, we further evaluate the frame-work's effectiveness and generalization capability through the KITTI benchmark test, and compare its performance with other published detection methods on the 3D detection benchmark and BEV detection benchmark for the 'Car' category, showing ex-cellent results.
  

# Installation
1. Clone this repository.
2. Our net is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), Please check [INSTALL.md](https://github.com/wangguojun2018/CenterNet3d/blob/master/docs/install.md) for installation instructions.

# Train
To train the CenterNet3D, run the following command:
```

python tools/train.py configs/mvxnet/test_dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
```

# Eval
To evaluate the model, run the following command:
```

python tools/test.py configs/mvxnet/test_dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py work_dirs/XXX/latest.pth --eval-options submission_prefix=work_dirs/XXX/test_submission --format-only
```

