# RTPSeg
RTPSeg: A Multi-Modality Dataset for LiDAR Point Cloud Semantic Segmentation Assisted with RGB-Thermal Images In Autonomous Driving, ISPRS j. 2026.

# Highlights
To bridge the research gap and promote in-depth research, we introduce RTPSeg, the first and only dataset comprising RGB and TIR images for 3D semantic segmentation in autonomous driving. RTPSeg fills the void of a specialized dataset and establishes itself as a novel benchmark, presenting considerable challenges and opportunities to existing approaches. More importantly, as a pioneering effort, the introduction of RTPSeg effectively validates the effectiveness of TIR images for 3D semantic segmentation, which has not been publicly validated before as far as we know. We hope that RTPSeg will spur in-depth exploration in this field.

To validate RTPSeg, we also propose RTPSegNet, a baseline model for 3D semantic segmentation assisted with RGB-thermal images, achieving the SOTA performance on RTPSeg and exhibiting promising effectiveness in jointly leveraging the complementary information between point clouds, RGB images, and TIR images.  Compared with previous multi-modality methods, RTPSegNet can solve the challenge of RGB images degeneration under backlighting or low-light conditions by integrating the thermal radiation information of TIR images into model training. 



# Demo
| Name | Vedio Demo |
|------|------|
| LiDAR_GT | <img src="./demo/output_video_lidar_gt.gif" width="300" alt="演示1"/> |
| RGB Images | <img src="./demo/output_video.gif" width="300" alt="演示2"/> |
| TIR Images| <img src="./demo/output_video_infra.gif" width="300" alt="演示3"/> |
| RGB Images with Projected Points | <img src="./demo/output_video_projection.gif" width="300" alt="演示4"/> |
| TIR Images with Projected Points | <img src="./demo/output_video_infra_projection.gif" width="300" alt="演示5"/> |

# Dataset
File shared via Baidu Netdisk: RTPSeg.zip
Link: https://pan.baidu.com/s/1Ms8XwyeOpHwP9RWQrDoBag
Access code: kb24

./dataset/

├── RTPSeg/

│   ├── 01

│   ├── 02

│   ├── ...

│   ├── 102

│   └── 103

%# Inference
We provide the pretrained model of ResNet and RTPSegNet for inference:

File shared via Baidu Netdisk: model
Link: https://pan.baidu.com/s/1fSFwQJUY7YDkVshANxMcaw
Access code: kb24

Note that we only provide the best version of RTPSegNet jointly trained with RGB and thermal infrared images, corresponding to the highest performance in our paper. You can choose to retrain the model to get other versions of RTPSegNet.

# Environment
You can refer to the requirements.txt to construct the experimental environment, or pip install -r requirements.txt.
We confirm that our project can be conducted on Python 3.8, CUDA 11.X or 12.X.

# Citation
If you find our work useful in your research, please consider citing:

@article{SUN202625,

author = {Yifan Sun and Chenguang Dai and Wenke Li and Xinpu Liu and Yongqi Sun and Ye Zhang and Weijun Guan and Yongsheng Zhang and Yulan Guo and Hanyun Wang},

title = {RTPSeg: A multi-modality dataset for LiDAR point cloud semantic segmentation assisted with RGB-thermal images in autonomous driving},

journal = {ISPRS Journal of Photogrammetry and Remote Sensing},

volume = {233},

pages = {25-38},

year = {2026},

issn = {0924-2716},

doi = {https://doi.org/10.1016/j.isprsjprs.2026.01.008},

url = {https://www.sciencedirect.com/science/article/pii/S0924271626000080},

keywords = {Dataset, Point cloud semantic segmentation, RGB image, Thermal infrared image, Autonomous driving},

}

# Acknowledgement
Thanks the following work, which give us much inspiration:

@inproceedings{yan20222dpass,

  title={2dpass: 2d priors assisted semantic segmentation on lidar point clouds},
  
  author={Yan, Xu and Gao, Jiantao and Zheng, Chaoda and Zheng, Chao and Zhang, Ruimao and Cui, Shuguang and Li, Zhen},
  
  booktitle={European Conference on Computer Vision},
  
  pages={677--695},
  
  year={2022},
  
  organization={Springer}
}

# Feedback
For any questions or feedback, please feel free to contact the author, who will make every effort to assist in order to ensure that this work truly serves community research.
