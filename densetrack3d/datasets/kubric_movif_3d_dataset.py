# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import imageio
import numpy as np
import torch
from densetrack3d.datasets.utils import DeltaData, aug_depth
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur


class BasicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(BasicDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.use_augs = use_augs
        self.crop_size = crop_size

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        gotit = False

        sample, gotit = self.getitem_helper(index)

        while not gotit:
            index = (index + 123) % len(self)
            sample, gotit = self.getitem_helper(index)
            # if not gotit:
            #     print("warning: sampling failed")
            #     # fake sample, so we can still collate
            #     sample = DeltaData(
            #         video=torch.zeros(
            #             (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
            #         ),
            #         videodepth=torch.zeros(
            #             (self.seq_len, 1, self.crop_size[0], self.crop_size[1])
            #         ),
            #         segmentation=torch.zeros(
            #             (self.seq_len, 1, self.crop_size[0], self.crop_size[1])
            #         ),
            #         trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
            #         visibility=torch.zeros((self.seq_len, self.traj_per_sample)),
            #         valid=torch.zeros((self.seq_len, self.traj_per_sample)),
            #     )

        return sample, gotit

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:  # eraser the specific region in the image
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(np.random.randint(1, self.eraser_max + 1)):  # number of times to occlude

                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:

            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max + 1)):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles, depths=None, intrs=None):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        if depths is not None:
            depths = [depth.astype(np.float32) for depth in depths]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        if depths is not None:
            # FIXME pad depth with edge value
            # depths = [
            #     np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)))
            #     for depth in depths
            # ]

            depths = [np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode="edge") for depth in depths]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0

        # transform intrinsic
        if intrs is not None:
            intrs[:, 0, 2] += pad_x0
            intrs[:, 1, 2] += pad_y0

        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        depths_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = scale_delta_x * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                scale_delta_y = scale_delta_y * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = W_new / float(W)
            scale_y = H_new / float(H)

            rgbs_scaled.append(cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            if depths is not None:
                depths_scaled.append(cv2.resize(depths[s], (W_new, H_new), interpolation=cv2.INTER_NEAREST))
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y

            # transform intrinsic
            if intrs is not None:
                intrs[s, 0, 2] *= scale_x
                intrs[s, 1, 2] *= scale_y
                intrs[s, 0, 0] *= scale_x
                intrs[s, 1, 1] *= scale_y

        rgbs = rgbs_scaled
        if depths is not None:
            depths = depths_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
                offset_y = int(
                    offset_y * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            if depths is not None:
                depths[s] = depths[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

            # transform intrinsic
            if intrs is not None:
                intrs[s, 0, 2] -= x0
                intrs[s, 1, 2] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
                if depths is not None:
                    depths = [depth[:, ::-1] for depth in depths]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                if depths is not None:
                    depths = [depth[::-1] for depth in depths]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]
        if depths is not None:
            return rgbs, trajs, depths, intrs
        else:
            return rgbs, trajs

    def crop(self, rgbs, trajs, depths=None):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else np.random.randint(0, H_new - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W_new else np.random.randint(0, W_new - self.crop_size[1])
        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]
        if depths is not None:
            depths = [depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for depth in depths]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0
        if depths is not None:
            return rgbs, trajs, depths
        else:
            return rgbs, trajs


class KubricMovifDataset(BasicDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
        use_wind_augs=False,
        use_video_flip=False,
    ):
        super(KubricMovifDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )
        self.intr = np.array([[560, 0, 256], [0, 560, 256], [0, 0, 1]])
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        # self.dataRootFrames = os.path.join(data_root, "512x512_frames")
        # self.dataRootDepth = os.path.join(data_root, "512x512_depth")

        self.seq_names = [fname for fname in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, fname))]

        # FIXME
        # if "small" in data_root:
        #     self.seq_names = self.seq_names[:5]
        self.use_wind_augs = use_wind_augs
        self.use_video_flip = use_video_flip
        print("found %d unique videos in %s" % (len(self.seq_names), data_root))

    def getitem_helper(self, index):

        # aug the sampling from 60 to 356
        # if self.use_wind_augs == True:
        #     self.traj_per_sample = torch.randint(64, 356, (1,)).item()

        gotit = True
        seq_name = self.seq_names[index]

        # npy_path = os.path.join(self.data_root, "512x512_anno", seq_name + ".npy")
        # rgb_path = os.path.join(self.dataRootFrames, seq_name)
        # depth_path = os.path.join(self.dataRootDepth, seq_name)

        npy_path = os.path.join(self.data_root, seq_name, seq_name + ".npy")
        rgb_path = os.path.join(self.data_root, seq_name, "frames")
        depth_path = os.path.join(self.data_root, seq_name, "depths")

        img_paths = sorted(os.listdir(rgb_path))
        # if self.use_video_flip:
        #     #NOTE: flip the video
        #     img_paths_inv = img_paths[::-1]
        #     img_paths = img_paths + img_paths_inv
        # loading the rgb frames
        rgbs = []
        intrs = []
        for i, img_path in enumerate(img_paths):
            if img_path.endswith(".npy"):
                continue
            rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))
            intrs.append(self.intr)
        rgbs = np.stack(rgbs)
        intrs = np.stack(intrs)
        # loading the depth frames
        dp_paths = sorted(os.listdir(depth_path))
        # if self.use_video_flip:
        #     #NOTE: flip the video
        #     dp_paths_inv = dp_paths[::-1]
        #     dp_paths = dp_paths + dp_paths_inv
        depths = []
        for i, dp_path in enumerate(dp_paths):
            depth16 = cv2.imread(os.path.join(depth_path, dp_path), cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32)
            depths.append(depth)
            # depths.append(imageio.v2.imread(os.path.join(depth_path, dp_path)))
        depths = np.stack(depths)[..., None]  # T, H, W, 1

        # print(depths)

        # print('prev deptrh', np.unique(depths))
        annot_dict = np.load(npy_path, allow_pickle=True).item()

        traj_2d = annot_dict["coords"]
        traj_depth = annot_dict["reproj_depth"].squeeze()[..., None]
        visibility = annot_dict["visibility"]
        depth_range = annot_dict["depth_range"].astype(float)
        # w2c = annot_dict["world_to_cam"] # T, 4, 4

        # print("depth_range", depth_range)
        # NOTE convert to metric depths (https://github.com/google-research/kubric/blob/7fa640a907a37bcea1c2e2f8c05ffdec6c90c6fc/challenges/point_tracking/dataset.py#L536)
        depth_min = depth_range[0]
        depth_max = depth_range[1]
        depth_f32 = depths.astype(float)

        # print("depth prev", depth_f32.max(), depth_f32.min())
        depths = depth_min + depth_f32 * (depth_max - depth_min) / 65535.0

        # print("depth post", depths.max(), depths.min())

        # depths = euclidean_to_image_plane_depth(depths.squeeze(-1), self.intr)[..., None]
        # depths = depths / 65535.0
        ##############################

        # print('depth_range', depths.max(), depths.min())

        # NOTE get depth
        # n_points, n_frames = traj_2d.shape[:2]
        # trak_2d_flatten = traj_2d.reshape(-1, 2)

        # traj_depth = []
        # for i in range(len(depths)):
        #     traj_depth_ = bilinear_interpolate_numpy(depths[i][..., 0], traj_2d[:, i, 0], traj_2d[:, i, 1])
        #     traj_depth.append(traj_depth_)
        # traj_depth = np.stack(traj_depth, axis=1)[..., None]

        # print('min traj_depth', traj_depth.min())
        # print('traj_depth', traj_depth.shape)
        # breakpoint()

        # traj_2d = annot_dict["target_points"].squeeze()
        # visibility = annot_dict["occluded"].squeeze()
        # traj_3d = annot_dict["CamCoordPos"].squeeze()

        # if self.use_video_flip:
        #     traj_2d_inv = traj_2d[:, ::-1, :]
        #     visibility_inv = visibility[:, ::-1]
        #     # traj_3d_inv = traj_3d[:, ::-1, :]
        #     traj_depth_inv = traj_depth[:, ::-1, :]

        #     traj_2d = np.concatenate((traj_2d, traj_2d_inv), axis=1)
        #     visibility = np.concatenate((visibility, visibility_inv), axis=1)
        #     # traj_3d = np.concatenate((traj_3d, traj_3d_inv), axis=1)
        #     traj_depth = np.concatenate((traj_depth, traj_depth_inv), axis=1)

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]
            rgbs = rgbs[start_ind : start_ind + self.seq_len]
            intrs = intrs[start_ind : start_ind + self.seq_len]
            depths = depths[start_ind : start_ind + self.seq_len]
            traj_2d = traj_2d[:, start_ind : start_ind + self.seq_len]
            traj_depth = traj_depth[:, start_ind : start_ind + self.seq_len]
            # traj_depth = traj_depth[:, start_ind : start_ind + self.seq_len]
            visibility = visibility[:, start_ind : start_ind + self.seq_len]

            # w2c = w2c[start_ind : start_ind + self.seq_len]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        # traj_3d = np.transpose(traj_3d, (1, 0, 2))
        traj_depth = np.transpose(traj_depth, (1, 0, 2))
        # merge the traj_3d into traj_2d
        # print(traj_2d.shape, traj_3d.shape)

        # print(traj_2d.shape)
        # breakpoint()
        # FIXME change to traj_depth
        # traj_2d = np.concatenate((traj_2d, traj_3d[..., 2:]), axis=2)
        traj_2d = np.concatenate((traj_2d, traj_depth), axis=2)

        visibility = np.transpose(np.logical_not(visibility), (1, 0))
        # print((~visibility).sum())
        # print((traj_3d[..., 2:]>0).sum())

        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(rgbs, traj_2d, visibility)

            # NOTE test disable spatial augs
            rgbs, traj_2d, depths, intrs = self.add_spatial_augs(rgbs, traj_2d, visibility, depths=depths, intrs=intrs)
        else:
            rgbs, traj_2d, depths = self.crop(rgbs, traj_2d, depths)
            depths = [depth[..., 0] for depth in depths]

        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)[:, 0]
            visibile_pts_inds = torch.cat((visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0)

        if len(visibile_pts_inds) >= self.traj_per_sample:
            point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        else:
            point_inds = np.random.choice(len(visibile_pts_inds), self.traj_per_sample, replace=True)
        # if len(point_inds) < self.traj_per_sample:
        #     gotit = False

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((self.seq_len, self.traj_per_sample))

        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()

        # # depths = torch.from_numpy(np.stack(depths)[...,None]/1000.0).permute(0,
        # #                                                         3, 1, 2).float()
        depths = torch.from_numpy(np.stack(depths)[..., None]).permute(0, 3, 1, 2).float()  # FIXME no scale depth

        # rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()
        # depths = torch.from_numpy(depths).permute(0, 3, 1, 2).float() # FIXME no scale depth
        intrs = torch.from_numpy(np.stack(intrs)).float()
        # w2c = torch.from_numpy(np.stack(w2c)).float()
        # add the scale augmentation for depth maps

        # FIXME disable augmentation depth
        if self.use_augs:
            depths = aug_depth(
                depths,
                grid=(8, 8),
                scale=(0.85, 1.15),
                shift=(-0.05, 0.05),
                gn_kernel=(7, 7),
                gn_sigma=(2, 2),
            )

        # if (depths > 0.01).sum() == 0:
        #     print(seq_name)
        # print("depth post", depths.max(), depths.min(), traj_2d[..., 2].min(), traj_2d[..., 2].max())

        segs = torch.ones((self.seq_len, 1, self.crop_size[0], self.crop_size[1]))
        sample = DeltaData(
            video=rgbs,
            videodepth=depths,
            segmentation=segs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            seq_name=seq_name,
            intrs=intrs,
        )
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)
