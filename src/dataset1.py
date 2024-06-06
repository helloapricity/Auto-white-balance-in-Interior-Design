from datasets import load_dataset
from torch.utils.data import IterableDataset
import logging
import torch
import numpy as np
import os.path as path
from src import ops
from DeepWB.arch import deep_wb_single_task as dwb
from DeepWB.utilities.deepWB import deep_wb
from DeepWB.utilities.utils import colorTempInterpolate_w_target

class StreamingData(IterableDataset):
    def __init__(self, dataset, patch_size=128, patch_number=32, aug=True,
                 wb_settings=None, shuffle_order=False, mode='training',
                 multiscale=False, keep_aspect_ratio=False, t_size=320):
        """ Data constructor """
        if wb_settings is None:
            self.wb_settings = ['D', 'T', 'F', 'C', 'S']
        else:
            self.wb_settings = wb_settings
        assert ('S' in self.wb_settings and 'T' in self.wb_settings and 'D' in self.wb_settings), 'Incorrect WB settings'

        for wb_setting in self.wb_settings:
            assert wb_setting in ['D', 'T', 'F', 'C', 'S']

        self.dataset = dataset
        self.patch_size = patch_size
        self.patch_number = patch_number
        self.keep_aspect_ratio = keep_aspect_ratio
        self.aug = aug
        self.multiscale = multiscale
        self.shuffle_order = shuffle_order
        assert (mode == 'training' or mode == 'testing'), 'mode should be training or testing'
        self.mode = mode

        if shuffle_order is True and self.mode == 'testing':
            logging.warning('Shuffling is not allowed in testing mode')
            self.shuffle_order = False

        self.t_size = t_size

        if self.mode == 'testing':
            self.deepWB_T = dwb.deepWBnet()
            self.deepWB_T.load_state_dict(torch.load('DeepWB/models/net_t.pth'))
            self.deepWB_S = dwb.deepWBnet()
            self.deepWB_S.load_state_dict(torch.load('DeepWB/models/net_s.pth'))
            self.deepWB_T.eval().to(device='cuda')
            self.deepWB_S.eval().to(device='cuda')

        logging.info(f'Creating streaming dataset')
    
    def __iter__(self):
        """ Return an iterator for the dataset. """
        for data in self.dataset:
            D_img = data['D_CS']
            S_img = data['S_CS']
            T_img = data['T_CS']
            GT_img = data['G_AS']

            base_name = 'image'  # Placeholder, you can replace it with actual logic if needed

            if self.mode == 'training':
                if self.multiscale:
                    t_size = self.t_size + 64 * 2 ** np.random.randint(5)
                else:
                    t_size = self.t_size

                D_img = ops.imresize.imresize(np.array(D_img), output_shape=(t_size, t_size))
                GT_img = ops.imresize.imresize(np.array(GT_img), output_shape=(t_size, t_size))
                S_img = ops.imresize.imresize(np.array(S_img), output_shape=(t_size, t_size))
                T_img = ops.imresize.imresize(np.array(T_img), output_shape=(t_size, t_size))

                F_img = data.get('F_CS', None)
                if F_img is not None:
                    F_img = ops.imresize.imresize(np.array(F_img), output_shape=(t_size, t_size))

                C_img = data.get('C_CS', None)
                if C_img is not None:
                    C_img = ops.imresize.imresize(np.array(C_img), output_shape=(t_size, t_size))

                if self.aug:
                    if F_img is not None and C_img is not None:
                        D_img, S_img, T_img, F_img, C_img, GT_img = ops.aug(
                            D_img, S_img, T_img, F_img, C_img, GT_img)
                    elif F_img is not None:
                        D_img, S_img, T_img, F_img, GT_img = ops.aug(
                            D_img, S_img, T_img, F_img, GT_img)
                    elif C_img is not None:
                        D_img, S_img, T_img, C_img, GT_img = ops.aug(
                            D_img, S_img, T_img, C_img, GT_img)
                    else:
                        D_img, S_img, T_img, GT_img = ops.aug(D_img, S_img, T_img, GT_img)

                if F_img is not None and C_img is not None:
                    D_img, S_img, T_img, F_img, C_img, GT_img = ops.extract_patch(
                        D_img, S_img, T_img, F_img, C_img, GT_img, patch_size=self.patch_size,
                        patch_number=self.patch_number)
                elif F_img is not None:
                    D_img, S_img, T_img, F_img, GT_img = ops.extract_patch(
                        D_img, S_img, T_img, F_img, GT_img, patch_size=self.patch_size,
                        patch_number=self.patch_number)
                elif C_img is not None:
                    D_img, S_img, T_img, C_img, GT_img = ops.extract_patch(
                        D_img, S_img, T_img, C_img, GT_img, patch_size=self.patch_size,
                        patch_number=self.patch_number)
                else:
                    D_img, S_img, T_img, GT_img = ops.extract_patch(
                        D_img, S_img, T_img, GT_img, patch_size=self.patch_size,
                        patch_number=self.patch_number)

                D_img = ops.to_tensor(D_img, dims=3 + int(self.aug)) / 255.0
                S_img = ops.to_tensor(S_img, dims=3 + int(self.aug)) / 255.0
                T_img = ops.to_tensor(T_img, dims=3 + int(self.aug)) / 255.0
                GT_img = ops.to_tensor(GT_img, dims=3 + int(self.aug)) / 255.0
                if F_img is not None:
                    F_img = ops.to_tensor(F_img, dims=3 + int(self.aug)) / 255.0
                if C_img is not None:
                    C_img = ops.to_tensor(C_img, dims=3 + int(self.aug)) / 255.0

                if self.shuffle_order:
                    imgs = [D_img, S_img, T_img]
                    if F_img is not None:
                        imgs.append(F_img)
                    if C_img is not None:
                        imgs.append(C_img)
                    order = np.random.permutation(len(imgs))

                    img = torch.cat((imgs[order[0]], imgs[order[1]], imgs[order[2]]), dim=1)
                    for i in range(3, len(imgs), 1):
                        img = torch.cat((img, imgs[order[i]]), dim=1)
                else:
                    img = torch.cat((D_img, S_img, T_img), dim=1)
                    if F_img is not None:
                        img = torch.cat((img, F_img), dim=1)
                    if C_img is not None:
                        img = torch.cat((img, C_img), dim=1)

                yield {'image': img, 'gt': GT_img, 'filename': base_name}

            else:  # testing mode
                paths = [S_img, T_img]
                F_img = data.get('F_CS', None)
                if F_img is not None:
                    paths.append(F_img)
                C_img = data.get('C_CS', None)
                if C_img is not None:
                    paths.append(C_img)

                if self.keep_aspect_ratio:
                    D_img = ops.aspect_ratio_imresize(np.array(D_img), max_output=self.t_size)
                else:
                    D_img = ops.imresize.imresize(np.array(D_img), output_shape=(self.t_size, self.t_size))
                
                for i in range(len(paths)):
                    if self.keep_aspect_ratio:
                        paths[i] = ops.aspect_ratio_imresize(np.array(paths[i]), max_output=self.t_size)
                    else:
                        paths[i] = ops.imresize.imresize(np.array(paths[i]), output_shape=(self.t_size, self.t_size))

                s_img = paths[0]
                t_img = paths[1]

                s_mapping = ops.get_mapping_func(D_img, s_img)
                full_size_s = ops.apply_mapping_func(D_img, s_mapping)
                full_size_s = ops.outOfGamutClipping(full_size_s)

                t_mapping = ops.get_mapping_func(D_img, t_img)
                full_size_t = ops.apply_mapping_func(D_img, t_mapping)
                full_size_t = ops.outOfGamutClipping(full_size_t)

                if len(paths) > 2:
                    f_img = paths[2]
                    f_mapping = ops.get_mapping_func(D_img, f_img)
                    full_size_f = ops.apply_mapping_func(D_img, f_mapping)
                    full_size_f = ops.outOfGamutClipping(full_size_f)
                else:
                    f_img = None

                if len(paths) > 3:
                    c_img = paths[3]
                    c_mapping = ops.get_mapping_func(D_img, c_img)
                    full_size_c = ops.apply_mapping_func(D_img, c_mapping)
                    full_size_c = ops.outOfGamutClipping(full_size_c)
                else:
                    c_img = None

                D_img = ops.to_tensor(D_img, dims=3) / 255.0
                S_img = ops.to_tensor(S_img, dims=3) / 255.0
                T_img = ops.to_tensor(T_img, dims=3) / 255.0

                if F_img is not None:
                    F_img = ops.to_tensor(F_img, dims=3) / 255.0
                if C_img is not None:
                    C_img = ops.to_tensor(C_img, dims=3) / 255.0

                img = torch.cat((D_img, S_img, T_img), dim=0)
                if F_img is not None:
                    img = torch.cat((img, F_img), dim=0)
                if C_img is not None:
                    img = torch.cat((img, C_img), dim=0)

                full_size_img = ops.to_tensor(full_size_img, dims=3) / 255.0
                full_size_s = ops.to_tensor(full_size_s, dims=3) / 255.0
                full_size_t = ops.to_tensor(full_size_t, dims=3) / 255.0

                if c_img is not None:
                    full_size_c = ops.to_tensor(full_size_c, dims=3) / 255.0

                if f_img is not None:
                    full_size_f = ops.to_tensor(full_size_f, dims=3) / 255.0

                if c_img is not None and f_img is not None:
                    yield {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
                        full_size_s, 'fs_t_img': full_size_t, 'fs_f_img': full_size_f,
                        'fs_c_img': full_size_c, 'filename': base_name}
                elif c_img is not None:
                    yield {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
                        full_size_s, 'fs_t_img': full_size_t, 'fs_c_img': full_size_c,
                        'filename': base_name}
                elif f_img is not None:
                    yield {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
                        full_size_s, 'fs_t_img': full_size_t, 'fs_f_img': full_size_f,
                        'filename': base_name}
                else:
                    yield {'image': img, 'fs_d_img': full_size_img, 'fs_s_img': full_size_s,
                        'fs_t_img': full_size_t, 'filename': base_name}

