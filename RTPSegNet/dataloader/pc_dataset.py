import os
import yaml
import numpy as np
#import open3d as o3d
from PIL import Image
from torch.utils import data

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))




@register_dataset
class RTPSeg(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            RTPSegyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = RTPSegyaml['learning_map']
        #self.learning_map_2 = RTPSegyaml['learning_map_2']
        self.imageset = imageset

        if imageset == 'train':

            #a = RTPSegyaml['split']['train'][0][0]
            #b = RTPSegyaml['split']['train'][0][1]
            #split = [x for x in range(a, b+1)]
            split = RTPSegyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += RTPSegyaml['split']['valid']
        elif imageset == 'val':
            #a = RTPSegyaml['split']['valid'][0][0]
            #b = RTPSegyaml['split']['valid'][0][1]
            #split = [x for x in range(a, b+1)]
            split = RTPSegyaml['split']['valid']
        elif imageset == 'test':
            split = RTPSegyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'lidar']), num_vote)
            #calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            #calib = self.read_calib(calib_path)
            #proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            #self.proj_matrix[i_folder] = proj_matrix

        #seg_num_per_class = config['dataset_params']['seg_labelweights']
        #seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        #self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def miu_rectify_points(points, miu_dict):
        R_LI = miu_dict['R_LI']
        R_out_inv = miu_dict['R_out_inv']
        R_end_inv = miu_dict['R_end_inv']
        T_LI = miu_dict['T_LI']
        T_out = miu_dict['T_out']
        T_end = miu_dict['T_end']
        for i in range(len(points)):
            point = points[i]
            points[i] = R_LI.T.dot(R_out_inv.dot(R_end_inv.dot(R_LI.dot(point) + T_LI) + T_end - T_out) - T_LI)
        return points


    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __getitem__(self, index):

        #cloud = PyntCloud.from_file(self.im_idx[index])
        #raw_data = cloud.points.to_numpy().squeeze()
        #self.im_idx[index]="/data/syf/dataset/pc_img_infra/datasets/04/lidar/27.npy"
        raw_data = np.load(self.im_idx[index])   #data0923 lidar is .npy format


        #pcd = o3d.io.read_point_cloud(self.im_idx[index])
        #points = np.asarray(pcd.points)
        #raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = raw_data.shape[0]
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
            #instance_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.load(self.im_idx[index].replace('lidar', 'labels')[:-3] + 'npz')['arr_0']#.reshape((-1, 1))
            annotated_data = annotated_data[:,1]
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            #annotated_data = np.vectorize(self.learning_map_2.__getitem__)(annotated_data)

            #instance_label = annotated_data[:,0]

        others_mask = annotated_data!=0
        annotated_data = annotated_data[others_mask] - 1
        points = points[others_mask]
        #annotated_data = annotated_data - 1
        #annotated_data[annotated_data==-1]=255
        image_file = self.im_idx[index].replace('lidar', 'camera').replace('.npy', '.png')
        image = Image.open(image_file)
        infra_file = self.im_idx[index].replace('lidar', 'infra').replace('.npy', '.png')
        infra = Image.open(infra_file)

        img_miu_path = self.im_idx[index].replace('lidar', 'imu_pos_rgb').replace('.npy', '.pkl')
        infra_miu_path = self.im_idx[index].replace('lidar', 'imu_pos_infra').replace('.npy', '.pkl')
        import pickle
        with open(img_miu_path, "rb") as f:
            img_miu_dict = pickle.load(f)
        with open(infra_miu_path, "rb") as f:
            infra_miu_dict = pickle.load(f)

        img_miu_points = self.miu_rectify_points(points, img_miu_dict)
        infra_miu_points =  self.miu_rectify_points(points, infra_miu_dict)


        data_dict = {}
        data_dict['xyz'] = points
        data_dict['img_miu_points'] = img_miu_points
        data_dict['infra_miu_points'] = infra_miu_points

        data_dict['labels'] = annotated_data.astype(np.uint8)
        #data_dict['instance_label'] = instance_label
        #data_dict['signal'] = raw_data[:, 3:4]
        data_dict['signal'] = raw_data[others_mask, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['infra'] = infra
        #data_dict['proj_matrix'] = proj_matrix
        data_dict['imageset'] = self.imageset

        return data_dict, self.im_idx[index]




def get_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        RTPSegyaml = yaml.safe_load(stream)
    label_name = dict()
    for i in sorted(list(RTPSegyaml['learning_map'].keys()))[::-1]:
        label_name[RTPSegyaml['learning_map'][i]] = RTPSegyaml['labels'][i]

    return label_name,RTPSegyaml['learning_map']




