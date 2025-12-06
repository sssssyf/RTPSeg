# RTPSeg
RTPSeg: A Multi-Modality Dataset for LiDAR Point Cloud Semantic Segmentation Assisted with RGB-Thermal Images

To bridge the research gap and promote in-depth research, we introduce RTPSeg, the first and only dataset comprising RGB and TIR images for 3D semantic segmentation in autonomous driving. RTPSeg fills the void of a specialized dataset and establishes itself as a novel benchmark, presenting considerable challenges and opportunities to existing approaches. More importantly, as a pioneering effort, the introduction of RTPSeg effectively validates the effectiveness of TIR images for 3D semantic segmentation, which has not been publicly validated before as far as we know. We hope that RTPSeg will spur in-depth exploration in this field.

To validate RTPSeg, we also propose RTPSegNet, a baseline model for 3D semantic segmentation assisted with RGB-thermal images, achieving the SOTA performance on RTPSeg and exhibiting promising effectiveness in jointly leveraging the complementary information between point clouds, RGB images, and TIR images.  Compared with previous multi-modality methods, RTPSegNet can solve the challenge of RGB images degeneration under backlighting or low-light conditions by integrating the thermal radiation information of TIR images into model training. 



| 功能 | 演示 |
|------|------|
| 安装过程 | <img src="./demo/output_video_lidar_gt.gif" width="300" /> |
| 使用示例 | <img src="./demo/output_video.gif" width="400" /> |
