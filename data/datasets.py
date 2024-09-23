import os
import torch
from torch.utils.data import Dataset
from .data_utils import pkload
import random
import numpy as np
import SimpleITK as sitk

class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
class OASISBrainValDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.img_paths = os.path.join(data_path, 'Val_img/')
        self.seg_paths = os.path.join(data_path, 'Val_seg/')
        self.number = 90
        self.transforms = transforms
        data_list = '/home/SLNet/OASIS/OASIS_L2R_2021_task03/val_pairs.txt'
        m_names = []
        f_names = []
        with open(os.path.join(self.img_paths, data_list), 'r') as f:
            for line in f.readlines():
                m_name, f_name = line.strip().split()
                m_names.append(m_name)
                f_names.append(f_name)
        self.m_names = m_names
        self.f_names = f_names
    def __getitem__(self, index):
        m_name = self.m_names[index]
        f_name = self.f_names[index]
        x = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.img_paths, m_name)))
        y = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.img_paths, f_name)))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.seg_paths, m_name)))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.seg_paths, f_name)))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x, y, x_seg, y_seg = x.permute(0,3,2,1), y.permute(0,3,2,1), x_seg.permute(0,3,2,1), y_seg.permute(0,3,2,1)
        return x, y, x_seg, y_seg

    def __len__(self):
        return self.number
class OASISBrainTestDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.img_paths = os.path.join(data_path, 'Test_img/')
        self.seg_paths = os.path.join(data_path, 'Test_seg/')
        self.number = 90
        self.transforms = transforms
        data_list = '/home/SLNet/OASIS/OASIS_L2R_2021_task03/test_pairs.txt'
        m_names = []
        f_names = []
        with open(os.path.join(self.img_paths, data_list), 'r') as f:
            for line in f.readlines():
                m_name, f_name = line.strip().split()
                m_names.append(m_name)
                f_names.append(f_name)
        self.m_names = m_names
        self.f_names = f_names
    def __getitem__(self, index):
        m_name = self.m_names[index]
        f_name = self.f_names[index]
        x = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.img_paths, m_name)))
        y = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.img_paths, f_name)))
        x_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.seg_paths, m_name)))
        y_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.seg_paths, f_name)))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x, y, x_seg, y_seg = x.permute(0,3,2,1), y.permute(0,3,2,1), x_seg.permute(0,3,2,1), y_seg.permute(0,3,2,1)
        return x, y, x_seg, y_seg

    def __len__(self):
        return self.number
