# RTPSeg
RTPSeg: A Multi-Modality Dataset for LiDAR Point Cloud Semantic Segmentation Assisted with RGB-Thermal Images In Autonomous Driving

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

# Dataset Dowload
File shared via Baidu Netdisk: RTPSeg.zip
Link: https://pan.baidu.com/s/1Ms8XwyeOpHwP9RWQrDoBag
Access code: kb24

# Cite
{Yifan Sun, Chenguang Dai, Wenke Li, Xinpu Liu, Yongqi Sun, Ye Zhang, Weijun Guan, Yongsheng Zhang, Yulan Guo, Hanyun Wang,
RTPSeg: A multi-modality dataset for LiDAR point cloud semantic segmentation assisted with RGB-thermal images in autonomous driving,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume 233,
2026,
Pages 25-38,
ISSN 0924-2716,
https://doi.org/10.1016/j.isprsjprs.2026.01.008}
