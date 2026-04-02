
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import json

def projection(json_path, pc_as_np, keep_idx, image=None, infra=False):

    # 读取相机内参
    json_file = open(json_path, "r")
    config_data = json.load(json_file)
    camera_matrix = np.array(config_data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(config_data["dist_coeffs"], dtype=np.float64)
    R = np.array(config_data["rotation_matrix"], dtype=np.float64)
    t = np.array(config_data["translation_vector"], dtype=np.float64)
    # 相机——雷达矩阵 -> 雷达——相机矩阵
    # R_inv = np.linalg.inv(R)
    # t = -R_inv.dot(t)
    # R = R_inv
    # global img1

    # camera_point = R.dot(pc_as_np.T) + t.reshape(3, 1)
    # d_mask = camera_point[2, :] >= 0.5
    # filter_points = pc_as_np[d_mask, :]

    projected_points = []
    count = 0
    # print("Src points length:", len(pc_as_np))
    pos=-1
    for point in pc_as_np:
        # 将点云坐标转换为齐次坐标
        # point_homogeneous = np.append(point, 1)
        pos+=1
        point = np.reshape(point, (3, 1))
        # 进行坐标变换
        transformed_point = np.dot(R, point) + t
        # 判断是否为相机后面的点
        if transformed_point[2] < 1:
            count += 1
            # print("delete point")
            continue
        # 进行透视投影
        projected_point_homogeneous = np.dot(camera_matrix, transformed_point)
        # 归一化
        projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
        # 添加到投影点列表中
        projected_points.append(projected_point)
        keep_idx[pos]=True


    # print("Deleted points length: ", count)
    # print("Projected points length: ", len(projected_points))
    # 转换为OpenCV格式的点
    projected_points = np.array([projected_points], dtype=np.float32)
    projected_points = projected_points.reshape(-1, 2)
    return projected_points, keep_idx




    '''
    if image is not None:
        if infra:
            image = np.array(image, dtype=np.float32, copy=False) / 255.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = np.array(image, dtype=np.float32, copy=False)[:,:,:3] / 255.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        for point in projected_points:
            x, y = point
            if (int(point[0]) > image.shape[1] or int(point[1]) > image.shape[0]
                    or int(point[0]) < 0 or int(point[1]) < 0):
                continue
            # cv2.circle(image, center=(int(point[0]), int(point[1])), radius=1, color=(int(point[0]), int(point[1]), 0), thickness=1)
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

        #cv2.imshow("Projected Point Cloud", image)
        #cv2.waitKey(0)
        #cv2.imwrite(json_path+'origin.png',image)
        #cv2.destroyAllWindows()
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite(json_path+'origin.jpg',image)
        if infra:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return projected_points, keep_idx, image
    else:
        return projected_points, keep_idx
    '''

REGISTERED_DATASET_CLASSES = {}
REGISTERED_COLATE_CLASSES = {}

try:
    from torchsparse import SparseTensor
    from torchsparse.utils.collate import sparse_collate_fn
    from torchsparse.utils.quantize import sparse_quantize
except:
    print('please install torchsparse if you want to run spvcnn/minkowskinet!')


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]




@register_dataset
class RTPSeg_dataset(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']


        self.image_json_path = config['dataset_params']['img_json_path']
        self.infra_json_path = config['dataset_params']['infra_json_path']
        self.image_json_path_2 = config['dataset_params']['img_json_path_2']
        self.infra_json_path_2 = config['dataset_params']['infra_json_path_2']
        self.infra_only = config['infra_only']
        self.image_only = config['image_only']
        self.image_and_infra = config['image_and_infra']
        self.voxel_size = config['model_params']['voxel_size']

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)



    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]
        root_list = root.split('/')
        sequence_num = int(root_list[-3])
        if sequence_num < 63:
            image_json_path = self.image_json_path
            infra_json_path = self.infra_json_path
        else:
            image_json_path = self.image_json_path_2
            infra_json_path = self.infra_json_path_2

        xyz = data['xyz']
        labels = data['labels'].reshape(-1,1)

        img_miu_points = data["img_miu_points"]
        infra_miu_points = data["infra_miu_points"]

        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])  #take a range of point cloud
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        ref_labels = ref_labels[mask]
        img_miu_points = img_miu_points[mask]
        infra_miu_points = infra_miu_points[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]


        point_num = len(xyz)


        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                ref_index[drop_idx] = ref_index[0]


        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        sig = (sig-np.mean(sig))/np.std(sig)   #density standardization
        #feat = np.concatenate((xyz, sig), axis=1)


        pc_ = np.round(xyz / self.voxel_size)
        pc_ = pc_ - pc_.min(0, keepdims=1)
        feat_ = np.concatenate((xyz, sig), axis=1)
        _, inds, inverse_map = sparse_quantize(pc_, 1, return_index=True, return_inverse=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels[inds]
        #img_miu_points = img_miu_points[inds]
        #infra_miu_points = infra_miu_points[inds]
        num_voxel = len(inds)
        points = SparseTensor(ref_pc, pc_)
        ref_index = SparseTensor(ref_index, pc_)
        map = SparseTensor(inds, pc)
        lidar = SparseTensor(feat, pc)
        labels_ = SparseTensor(labels, pc)
        labels_mapped = SparseTensor(ref_labels, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)



        data_dict = {}
        data_dict['lidar'] = lidar
        data_dict['points'] = points
        data_dict['targets'] = labels_
        data_dict['targets_mapped'] = labels_mapped
        data_dict['ref_index'] = ref_index
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root
        data_dict['map'] = map
        data_dict['num_voxel'] = num_voxel
        data_dict['point_num'] = point_num
        data_dict['inverse_map'] = inverse_map




        infra = data['infra']
        image = data['img']


        keep_idx_img = np.zeros(ref_labels.shape[0]).astype(bool)
        # project points into image
        img_points, keep_idx_img = projection(image_json_path, img_miu_points, keep_idx_img, image, infra=False)
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, image.size[0],image.size[1])  # Image_W, Image_H)
        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]
        keep_idx_img[keep_idx_img] = keep_idx_img_pts
        img_label = ref_labels[keep_idx_img]
        point2img_index = np.arange(ref_labels.shape[0])[keep_idx_img]


        keep_idx_infra = np.zeros(ref_labels.shape[0]).astype(bool)
        # project points into image
        infra_points, keep_idx_infra = projection(infra_json_path, infra_miu_points, keep_idx_infra, infra, infra=True)
        keep_idx_infra_pts = self.select_points_in_frustum(infra_points, 0, 0, infra.size[0], infra.size[1])
        infra_points = np.fliplr(infra_points)
        points_infra = infra_points[keep_idx_infra_pts]
        keep_idx_infra[keep_idx_infra] = keep_idx_infra_pts
        infra_label = ref_labels[keep_idx_infra]
        point2infra_index = np.arange(ref_labels.shape[0])[keep_idx_infra]






        ###  image crop ###
        if self.bottom_crop and self.image_only and data['imageset'] == 'train':
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            bottom = int(np.random.rand() * (image.size[1] + 1 - self.bottom_crop[1]))
            top = bottom + self.bottom_crop[1]

            # update image points     #480*320
            keep_idx = points_img[:, 0] >= bottom
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < top)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, bottom, right, top))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= bottom
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        ###  infra crop ###
        if self.bottom_crop and self.infra_only and data['imageset'] == 'train':
            left = int(np.random.rand() * (infra.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            bottom = int(np.random.rand() * (infra.size[1] + 1 - self.bottom_crop[1]))
            top = bottom + self.bottom_crop[1]

            # update image points     #480*320
            keep_idx = points_infra[:, 0] >= bottom
            keep_idx = np.logical_and(keep_idx, points_infra[:, 0] < top)
            keep_idx = np.logical_and(keep_idx, points_infra[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_infra[:, 1] < right)

            # crop image
            infra = infra.crop((left, bottom, right, top))
            points_infra = points_infra[keep_idx]
            points_infra[:, 0] -= bottom
            points_infra[:, 1] -= left

            infra_label = infra_label[keep_idx]
            point2infra_index = point2infra_index[keep_idx]

            img_label = infra_label
            point2img_index = point2infra_index


        if self.image_and_infra and data['imageset'] == 'train':  #get the overlapping points

            point2img_index_mask = np.zeros((point2img_index.shape[0])).astype(bool)
            point2infra_index_mask = np.zeros((point2infra_index.shape[0])).astype(bool)

            if point2img_index.shape[0] >= point2infra_index.shape[0]:
                for i, j in enumerate(point2img_index):
                    if j in point2infra_index:
                        point2img_index_mask[i] = True
                        point2infra_index_mask[j == point2infra_index] = True
            else:
                for i, j in enumerate(point2infra_index):
                    if j in point2img_index:
                        point2infra_index_mask[i] = True
                        point2img_index_mask[j == point2img_index] = True

            points_img = points_img[point2img_index_mask,:]
            img_label = img_label[point2img_index_mask]
            point2img_index = point2img_index[point2img_index_mask]

            try:
                img_range_w = points_img[:, 1].max()
            except ValueError:
                print("ValueError ")
                print(root)
                print("point2img_index_mask: ")
                print(point2img_index_mask)
                print("point2infra_index_mask: ")
                print(point2infra_index_mask)
                img_range_w = image.size[0]

            try:
                img_range_h = points_img[:, 0].max()
            except ValueError:
                print("ValueError")
                img_range_h = image.size[1]
            #img_range_w = points_img[:, 1].max()
            #img_range_h = points_img[:, 0].max()

            points_infra = points_infra[point2infra_index_mask,:]
            infra_label = infra_label[point2infra_index_mask]
            point2infra_index = point2infra_index[point2infra_index_mask]

            try:
                infra_range_w = points_img[:, 1].max()  # 空数组时返回 0
            except ValueError:
                print("ValueError")
                infra_range_w = infra.size[0]

            try:
                infra_range_h = points_img[:, 0].max()
            except ValueError:
                print("ValueError")
                infra_range_h = infra.size[1]

            #infra_range_w = points_infra[:, 1].max()
            #infra_range_h = points_infra[:, 0].max()

            image_random_crop_judge =True
            if self.bottom_crop:
                while image_random_crop_judge == True:

                    img_crop_range_w = int(img_range_w) + 1
                    img_crop_range_h = int(img_range_h) + 1
                    left = int(np.random.rand() * (img_crop_range_w - self.bottom_crop[0]))
                    right = left + self.bottom_crop[0]

                    # bottom = image_ori.size[1] - self.bottom_crop[1]
                    # top = image_ori.size[1]
                    bottom = int(np.random.rand() * (img_crop_range_h - self.bottom_crop[1]))
                    top = bottom + self.bottom_crop[1]

                    # update image points     #480*320
                    keep_idx = points_img[:, 0] >= bottom
                    keep_idx = np.logical_and(keep_idx, points_img[:, 0] < top)
                    keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
                    keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

                    # crop image

                    point2img_index_temp = point2img_index[keep_idx]

                    #if len(point2img_index_temp) > 1000:

                    image_tmp = image.crop((left, bottom, right, top))
                    points_img_temp = points_img[keep_idx]
                    points_img_temp[:, 0] -= bottom
                    points_img_temp[:, 1] -= left

                    img_label_temp = img_label[keep_idx]
                    # point2img_index = point2img_index[keep_idx]

                    infra_random_crop = 0
                    infra_random_crop_judge = True
                    while infra_random_crop < 5 and infra_random_crop_judge == True:
                        infra_random_crop += 1
                        # print("infra_random_crop:{}".format(infra_random_crop))

                        infra_crop_range_w = int(infra_range_w) + 1
                        infra_crop_range_h = int(infra_range_h) + 1

                        left = int(np.random.rand() * (infra_crop_range_w - self.bottom_crop[0]))
                        right = left + self.bottom_crop[0]
                        # bottom = image_ori.size[1] - self.bottom_crop[1]
                        # top = image_ori.size[1]
                        bottom = int(np.random.rand() * (infra_crop_range_h - self.bottom_crop[1]))
                        top = bottom + self.bottom_crop[1]

                        # update image points     #480*320
                        keep_idx = points_infra[:, 0] >= bottom
                        keep_idx = np.logical_and(keep_idx, points_infra[:, 0] < top)
                        keep_idx = np.logical_and(keep_idx, points_infra[:, 1] >= left)
                        keep_idx = np.logical_and(keep_idx, points_infra[:, 1] < right)

                        # crop image
                        point2infra_index_temp = point2infra_index[keep_idx]

                        comon_index = np.intersect1d(point2img_index_temp, point2infra_index_temp)
                        if point2img_index_temp.shape[0] == 0:
                            common_ratio = 0
                        else:
                            common_ratio = len(comon_index) / point2img_index_temp.shape[0]

                        if common_ratio > 0.7:
                            image = image_tmp

                            points_img = points_img_temp
                            img_label = img_label_temp
                            point2img_index = point2img_index_temp

                            infra = infra.crop((left, bottom, right, top))
                            points_infra = points_infra[keep_idx]
                            points_infra[:, 0] -= bottom
                            points_infra[:, 1] -= left

                            infra_label = infra_label[keep_idx]
                            point2infra_index = point2infra_index[keep_idx]

                            point2img_index_mask = np.zeros((point2img_index.shape[0])).astype(bool)
                            point2infra_index_mask = np.zeros((point2infra_index.shape[0])).astype(bool)

                            if point2img_index.shape[0] >= point2infra_index.shape[0]:
                                for i, j in enumerate(point2img_index):
                                    if j in point2infra_index:
                                        point2img_index_mask[i] = True
                                        point2infra_index_mask[j == point2infra_index] = True
                            else:
                                for i, j in enumerate(point2infra_index):
                                    if j in point2img_index:
                                        point2infra_index_mask[i] = True
                                        point2img_index_mask[j == point2img_index] = True

                            points_img = points_img[point2img_index_mask, :]
                            img_label = img_label[point2img_index_mask]
                            point2img_index = point2img_index[point2img_index_mask]

                            points_infra = points_infra[point2infra_index_mask, :]
                            infra_label = infra_label[point2infra_index_mask]
                            point2infra_index = point2infra_index[point2infra_index_mask]

                            image_random_crop_judge = False
                            infra_random_crop_judge = False

                            # point2img_index_mask2 = np.zeros((ref_labels.shape[0])).astype(np.int64)
                            # # img_label_mask = np.zeros((ref_labels.shape[0])).reshape(-1,1)
                            # for i in range(point2img_index.shape[0]):
                            #     point2img_index_mask2[point2img_index[i]] = point2infra_index[i]
                            #     # img_label_mask[point2img_index[i]]= img_label[i]
                            #
                            # point2img_index_mask2 = point2img_index_mask2[inds]
                            # others_mask = point2img_index_mask2 != 0
                            # point2img_index = point2img_index_mask2[others_mask]
                            # img_label = labels[others_mask]
                            # points_img_temp = np.zeros((point2img_index.shape[0], 2)).astype(np.float32)
                            # points_infra_temp = np.zeros((point2img_index.shape[0], 2)).astype(np.float32)
                            # for j in range(point2img_index.shape[0]):
                            #     index = point2img_index[j]
                            #     points_img_temp[j, :] = points_img[point2infra_index == index, :]
                            #     points_infra_temp[j, :] = points_infra[point2infra_index == index, :]
                            #
                            # points_img = points_img_temp
                            # points_infra = points_infra_temp



        img_indices = points_img.astype(np.int64)
        infra_indices = points_infra.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image = np.array(image, dtype=np.float32, copy=False)[:,:,:3] / 255.
        infra = np.array(infra, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:

            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
            infra = (infra - np.mean(infra))/ np.std(infra)

        # image_ref = np.array(image_ref, dtype=np.float32, copy=False)[:,:,:3] / 255.
        # infra_ref = np.array(infra_ref, dtype=np.float32, copy=False) / 255.

        # data_dict = {}
        # data_dict['point_feat'] = feat
        # data_dict['point_label'] = labels#.reshape(-1,1)
        # data_dict['ref_xyz'] = ref_pc
        # data_dict['ref_label'] = ref_labels#.reshape(-1,1)
        # data_dict['ref_index'] = ref_index
        # #data_dict['mask'] = mask
        # data_dict['point_num'] = point_num
        # data_dict['origin_len'] = origin_len
        # data_dict['root'] = root

        data_dict_2D = {}
        data_dict_2D['img'] = image
        data_dict_2D['img_indices'] = img_indices
        data_dict_2D['infra'] = infra.reshape(infra.shape[0],infra.shape[1],1)
        data_dict_2D['infra_indices'] = infra_indices
        data_dict_2D['img_label'] = img_label.reshape(-1,1)
        data_dict_2D['point2img_index'] = point2img_index
        data_dict.update(data_dict_2D)
        return data_dict





@register_collate_fn
def collate_fn_voxel_2D(inputs):


    num_voxel = [d['num_voxel'] for d in inputs]
    point_num = [d['point_num'] for d in inputs]
    batch_size = len(num_voxel)
    batch_size_2 = len(point_num)
    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(num_voxel[i]) * i)
    b_idx2 = []
    for i in range(batch_size_2):
        b_idx2.append(torch.ones(point_num[i]) * i)

    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in inputs]
    img = [torch.from_numpy(d['img']) for d in inputs]
    img_indices = [d['img_indices'] for d in inputs]
    infra = [torch.from_numpy(d['infra']) for d in inputs]
    infra_indices = [d['infra_indices'] for d in inputs]
    img_label = [torch.from_numpy(d['img_label']) for d in inputs]

    inputs2_back = {
        'batch_idx': torch.cat(b_idx).long(),
        'batch_idx_2': torch.cat(b_idx2).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'img_indices': img_indices,
        'infra': torch.stack(infra, 0).permute(0, 3, 1, 2),
        'infra_indices': infra_indices,
        'img_label': torch.cat(img_label, 0).squeeze(1).long(),
        }

    for d in inputs:
        del d['num_voxel']
        del d['point_num']
        del d['point2img_index']
        del d['img']
        del d['img_indices']
        del d['infra']
        del d['infra_indices']
        del d['img_label']

    inputs1_back = sparse_collate_fn(inputs)
    inputs1_back.update(inputs2_back)
    return inputs1_back
