import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import json
import numpy as np
from PIL import Image

def get_valid_stride(train_indices, test_indices, video_length):
    # Try up to some limit of attempts
    for attempt in range(10):
        stride = random.randint(1, 5)
        s_train = train_indices[::stride]
        s_test  = test_indices[::stride]

        # 1) Make sure you have enough total frames
        if len(s_train) + len(s_test) < video_length:
            # Not enough frames in total
            continue
        # 2) Make sure you have at least 2 train frames, if needed
        if len(s_train) < 2:
            continue
        # 3) Test frames also should not be 0 if your logic requires some
        if len(s_test) < video_length // 2:
            continue
        return stride
    return 1
    
def selected_close_indices(selected_train_indices, 
                           selected_test_indices, 
                           video_length, 
                           stride):
    debug_mode = False # HARDCODED
    # 1) Randomly pick how many train frames
    num_train_frames = random.randint(2, min(len(selected_train_indices), video_length // 2))
    num_test_frames  = video_length - num_train_frames

    max_train_idx = max(selected_train_indices)
    min_train_idx = min(selected_train_indices)
    max_test_idx  = max(selected_test_indices)
    min_test_idx  = min(selected_test_indices)

    # 2) Non-overlap cases
    # ------------------------------------------------------------
    # Case A: All train indices < all test indices
    if max_train_idx < min_test_idx:
        selected_indices = ( selected_train_indices[-num_train_frames:] +
                             selected_test_indices[:num_test_frames] )
        if debug_mode:
            print(f"[DEBUG] : 1-1 selected_indices = {selected_indices}")

    # Case B: All test indices < all train indices
    elif min_train_idx > max_test_idx:
        selected_indices = ( selected_test_indices[-num_test_frames:] +
                             selected_train_indices[:num_train_frames] )
        if debug_mode:
            print(f"[DEBUG] : 2-1 selected_indices = {selected_indices}")

    # 3) Overlap case: (some test indices on both sides, train in the middle)
    else:
        # split test indices into "left" and "right" of the train chunk
        left_test_indices  = [x for x in selected_test_indices if x < min_train_idx]
        right_test_indices = [x for x in selected_test_indices if x > max_train_idx]
        left_size  = len(left_test_indices)
        right_size = len(right_test_indices)

        # 3A) If left has more test indices, pick from left first, then right
        if left_size > right_size:
            # how many we can pick from left side
            left_subset_size = min(left_size, num_test_frames)
            # how many we still need from the right
            right_subset_size = min(num_test_frames - left_subset_size, right_size)
            num_train_frames = video_length - left_subset_size - right_subset_size
            # pick frames
            left_subset  = left_test_indices[-left_subset_size:]  # last N from the left
            train_subset = selected_train_indices[:num_train_frames]
            right_subset = right_test_indices[:right_subset_size] if right_subset_size > 0 else [] # first M from the right

            selected_indices = left_subset + train_subset + right_subset
            if debug_mode:
                print(f"[DEBUG] : 3-1 left_subset_size = {left_subset_size}, right_subset_size = {right_subset_size}, num_train_frames = {num_train_frames}")
                print(f"[DEBUG] : 3-1 left={left_subset_size}, right={right_subset_size}, "
                      f"train={num_train_frames}, total={len(selected_indices)} => {selected_indices}")

        # 3B) If right has more (or equal) test indices, pick from right first, then left
        else:
            right_subset_size = min(right_size, num_test_frames)
            left_subset_size  = min(num_test_frames - right_subset_size, left_size)
            num_train_frames = video_length - left_subset_size - right_subset_size
            left_subset  = left_test_indices[-left_subset_size:] if left_subset_size > 0 else []
            train_subset = selected_train_indices[-num_train_frames:]
            right_subset = right_test_indices[:right_subset_size]

            # pick train frames in the middle, or whichever order you want:
            selected_indices = left_subset + train_subset + right_subset

            if debug_mode:
                print(f"[DEBUG] : 3-2 left_subset_size = {left_subset_size}, right_subset_size = {right_subset_size}, num_train_frames = {num_train_frames}")
                print(f"[DEBUG] : 3-2 left={left_subset_size}, right={right_subset_size}, "
                      f"train={num_train_frames}, total={len(selected_indices)} => {selected_indices}")

    # 4) Final check for mismatch
    if len(selected_indices) != video_length:
        print(f"[WARNING] : len(selected_indices) != video_length, "
                         f"{len(selected_indices)} != {video_length}")
        if debug_mode:
            exit()
    
    return sorted(selected_indices)

class GSDataDL3DV(Dataset):
    """
    GSDataDL3DV Dataset.
    Assumes GSDataDL3DV data is structured as follows.
    GSDataDL3DV/
    """
    def __init__(self,
                 data_dir,
                 caption_file_name,
                 subsets,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 guidance_type='none',
                 relative_pose=False,
                 skip_subset=False # for RE10K
                 ):
        self.data_dir = data_dir
        self.subsets = subsets
        self.caption_file_name = caption_file_name
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.guidance_type = guidance_type
        self.relative_pose = relative_pose
        self.skip_subset = skip_subset

        self._load_metadata()
        self._load_scene_mappings()

        # Define spatial transformations if any
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                # self.spatial_transform = transforms.RandomCrop(self.resolution)
                # self.spatial_transform = RandomCrop(self.resolution)
                self.spatial_transform = "random_crop"
            elif spatial_transform == "resize_random_crop":
                self.spatial_transform = "resize_random_crop"
            # elif spatial_transform == "center_crop":
            #     self.spatial_transform = transforms.Compose([
            #         transforms.CenterCrop(resolution),
            #     ])            
            # elif spatial_transform == "resize_center_crop":
            #     self.spatial_transform = transforms.Compose([
            #         transforms.Resize(min(self.resolution)),
            #         transforms.CenterCrop(self.resolution),
            #     ])
            # elif spatial_transform == "resize":
            #     self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        """
        json file
        {"caption": [
            {"scene_id1": scene_id1,
             "caption1": caption1},
            {"scene_id2": scene_id2,
             "caption2": caption2},
            ...
        ]}
        """
        self.metadata = []
        
        if self.skip_subset:
            caption_path = os.path.join(self.data_dir, self.caption_file_name)
            with open(caption_path, 'r') as f:
                captions = json.load(f)
            self.metadata += captions["captions"]
            print(f'{self.data_dir} >>> {len(self.metadata)} data samples loaded.')
        else:
            for subset in self.subsets:
                caption_path = os.path.join(self.data_dir, subset, self.caption_file_name)
                with open(caption_path, 'r') as f:
                    captions = json.load(f)
                self.metadata += captions["captions"]
                print(f'{self.data_dir}/{subset} >>> {len(self.metadata)} data samples loaded.')

    def _load_scene_mappings(self):
        """
        Build a mapping from indices to (folder, filename) for each scene.
        """
        self.scene_mappings = {}
        for sample in tqdm(self.metadata):
            scene_id = sample['scene_id']
            subset = sample['subset'] if not self.skip_subset else ""
            
            if scene_id not in self.scene_mappings:
                scene_path = os.path.join(self.data_dir, subset, scene_id)
                # Load partition.json
                partition_path = os.path.join(scene_path, 'partition.json')
                with open(partition_path, 'r') as f:
                    partition = json.load(f)
                train_indices = partition['train']
                test_indices = partition['test']
                # Build index_to_file mapping for this scene
                index_to_file = {}
                # For train indices
                for i, idx in enumerate(train_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('train', filename)
                # For test indices
                for i, idx in enumerate(test_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('test', filename)
                # Store mapping
                self.scene_mappings[scene_id] = index_to_file

    def _get_scene_path(self, sample):
        if self.skip_subset:
            scene_path = os.path.join(self.data_dir, sample['scene_id'])
        else:
            scene_path = os.path.join(self.data_dir, sample['subset'], sample['scene_id'])
        return scene_path

    def __getitem__(self, index):
        try:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            scene_id = sample['scene_id']
            scene_path = self._get_scene_path(sample)
            index_to_file = self.scene_mappings[scene_id]

            # Get train and test indices
            train_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'train']
            test_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'test']

            train_indices.sort()
            test_indices.sort()

            # Randomly pick stride between 1 and 5
            stride = get_valid_stride(train_indices, test_indices, self.video_length)
            selected_train_indices = sorted(train_indices[::stride])
            selected_test_indices = sorted(test_indices[::stride])
            
            selected_indices = selected_close_indices(selected_train_indices, selected_test_indices, 
                                                    self.video_length, stride)

            if len(selected_indices) > self.video_length:
                selected_indices = selected_indices[:self.video_length]
            elif len(selected_indices) < self.video_length:
                repeats = (self.video_length + len(selected_indices) - 1) // len(selected_indices)
                selected_indices = (selected_indices * repeats)[:self.video_length]
            selected_indices.sort()

            crop_params = None
            # Load images and apply transformations
            images = []
            renders = []
            guidances = []
            cond_frame_idx = []
            for i, idx in enumerate(selected_indices):
                folder, filename = index_to_file[idx]
                if folder == 'train':
                    cond_frame_idx.append(i)
                image_path = os.path.join(scene_path, folder, 'ours_30000/gt', filename)
                render_path = os.path.join(scene_path, folder, 'ours_30000/renders', filename)
                image = Image.open(image_path).convert('RGB')
                render = Image.open(render_path).convert('RGB')
                # if self.spatial_transform is not None:
                #     image = self.spatial_transform(image)
                #     render = self.spatial_transform(render)
                # load guidances
                if self.guidance_type != 'none':
                    guidance_file_name = f"frame_{idx+1:05d}.png"
                    if self.guidance_type == 'visibility_mask_global_2ndpick_th0.5':
                        guidance_path = os.path.join(scene_path, folder, f'ours_30000/{self.guidance_type}', guidance_file_name)
                        guidance = Image.open(guidance_path).convert('L') # grayscale
                    else:
                        raise NotImplementedError(f"[ERROR] : Guidance type not supported {self.guidance_type}")
                else:
                    guidance = None

                if self.spatial_transform == "resize_random_crop":
                    image = transforms.Resize(max(self.resolution))(image)
                    render = transforms.Resize(max(self.resolution))(render)
                    if guidance is not None:
                        guidance = transforms.Resize(max(self.resolution))(guidance)
                
                if crop_params is None:
                    if self.spatial_transform == "random_crop" or self.spatial_transform == "resize_random_crop":
                        dummy = torch.zeros(3, image.size[1], image.size[0]) # get resolution from image (PIL)
                        crop_params = transforms.RandomCrop.get_params(dummy, self.resolution)
                
                if self.spatial_transform == "random_crop" or self.spatial_transform == "resize_random_crop":
                    # Sample crop params from the first image
                    a,b,c,d = crop_params
                    image = F.crop(image, a, b, c, d)
                    render = F.crop(render, a, b, c, d)
                    # Apply to guidance
                    if guidance is not None:
                        guidance = F.crop(guidance, a, b, c, d)
                    image = transforms.ToTensor()(image)
                    render = transforms.ToTensor()(render)
                    if guidance is not None:
                        guidance = transforms.ToTensor()(guidance)
                images.append(image)
                renders.append(render)
                if guidance is not None:
                    guidances.append(guidance)
                
            # Stack images into a tensor [C, T, H, W]
            frames = torch.stack(images, dim=1)
            renders = torch.stack(renders, dim=1)
            cond_frame_idx = random.choice(cond_frame_idx)
            if self.guidance_type != 'none':
                guidances = torch.stack(guidances, dim=1)

            # Load extrinsics and select corresponding ones
            cam_extrinsics = np.load(os.path.join(scene_path, 'cam_extrinsics.npy'))

            if self.relative_pose:
                # (1) normalization 
                # get center and diag of the scene
                center = np.mean(cam_extrinsics[:, :3, 3], axis=0)
                diag = np.max(np.linalg.norm(cam_extrinsics[:, :3, 3] - center, axis=1))
                # get relative pose
                cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] - center
                cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] / diag
                selected_extrinsics = cam_extrinsics[selected_indices]
                
                # (2) get relative pose
                criterion = cam_extrinsics[selected_train_indices][0]
                criterion_inv = np.linalg.inv(criterion)
                selected_extrinsics = np.einsum('ij,kjl->kil', criterion_inv, selected_extrinsics)
            else:
                selected_extrinsics = cam_extrinsics[selected_indices]
            selected_extrinsics = selected_extrinsics[:, :3, :].reshape(-1, 12)
            selected_extrinsics = torch.tensor(selected_extrinsics)
            # Prepare data dictionary
            data = {
                'video': frames,  # [C, T, H, W]
                'renders': renders,  # [C, T, H, W]
                'caption': sample['caption'],
                'camera_pose': selected_extrinsics,  # [T, ...]
                'path': scene_path,
                'fps': 5, # TODO : change it to use extrinsics
                'scene_id': scene_id,
                'cond_frame_idx': cond_frame_idx
            }
            if self.guidance_type != 'none':
                data['guidance'] = guidances
            # iterate data and print shape if value is tensor
            # for key, value in data.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"[DEBUG] : {key} shape = {value.shape}")
            #     else:
            #         print(f"[DEBUG] : {key} = {value}")
            return data
        except:
            return self.__getitem__(random.randint(0, len(self.metadata)-1))

        # TODO: merge seamlessly ...
        # if self.relative_pose:
        #     # (1) normalization 
        #     # get center and diag of the scene
        #     center = np.mean(cam_extrinsics[:, :3, 3], axis=0)
        #     diag = np.max(np.linalg.norm(cam_extrinsics[:, :3, 3] - center, axis=1))
        #     # get relative pose
        #     cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] - center
        #     cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] / diag
        #     selected_extrinsics = cam_extrinsics[selected_indices]
            
        #     # (2) get relative pose
        #     criterion = cam_extrinsics[selected_train_indices][0]
        #     criterion_inv = np.linalg.inv(criterion)
        #     selected_extrinsics = np.einsum('ij,kjl->kil', criterion_inv, selected_extrinsics)
        # else:
        #     selected_extrinsics = cam_extrinsics[selected_indices]
        # selected_extrinsics = selected_extrinsics[:, :3, :].reshape(-1, 12)
        # selected_extrinsics = torch.tensor(selected_extrinsics)
                
        # dino_feats_pca = torch.load(os.path.join(scene_path, 'dino_feats_pca.pth'))
        # # dino_feats = torch.load(os.path.join(scene_path, 'dino_feats_all.pth'))
        # control = dino_feats_pca[selected_indices]
        # control = control.permute(0, 3, 1, 2) # (T, H, W, 3) -> (T, 3, H, W)
        
        # # resize to H, W
        # control = torch.nn.functional.interpolate(control, size=(self.resolution[0], self.resolution[1]), mode='bilinear')
        
        # # Prepare data dictionary
        # data = {
        #     'video': frames,  # [C, T, H, W]
        #     'renders': renders,  # [C, T, H, W]
        #     'caption': sample['caption'],
        #     'camera_pose': selected_extrinsics,  # [T, ...]
        #     'path': scene_path,
        #     'fps': 5, # TODO : change it to use extrinsics
        #     'scene_id': scene_id,
        #     'cond_frame_idx': cond_frame_idx,
        #     "control": control
        # }
        # if self.guidance_type != 'none':
        #     data['guidance'] = guidances
        # # iterate data and print shape if value is tensor
        # # for key, value in data.items():
        # #     if isinstance(value, torch.Tensor):
        # #         print(f"[DEBUG] : {key} shape = {value.shape}")
        # #     else:
        # #         print(f"[DEBUG] : {key} = {value}")
        # return data
        # except:
        #     print(f"[ERROR] : DL3DV {index} is not valid")
        #     return self.__getitem__(random.randint(0, len(self.metadata)-1))

    def __len__(self):
        return len(self.metadata)

class GSDataDL3DV_latent(Dataset):
    """
    GSDataDL3DV Dataset.
    Assumes GSDataDL3DV data is structured as follows.
    GSDataDL3DV/
    """
    def __init__(self,
                 data_dir,
                 caption_file_name,
                 subsets,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 guidance_type='none',
                 relative_pose=False,
                 ):
        self.data_dir = data_dir
        self.subsets = subsets
        self.caption_file_name = caption_file_name
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.guidance_type = guidance_type
        self.relative_pose = relative_pose

        self._load_metadata()
        self._load_scene_mappings()

        # Define spatial transformations if any
        spatial_transform = "resize_center_crop" # enforce
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                # self.spatial_transform = transforms.RandomCrop(self.resolution)
                # self.spatial_transform = RandomCrop(self.resolution)
                self.spatial_transform = "random_crop"
            elif spatial_transform == "resize_random_crop":
                self.spatial_transform = "resize_random_crop"
            # elif spatial_transform == "center_crop":
            #     self.spatial_transform = transforms.Compose([
            #         transforms.CenterCrop(resolution),
            #     ])            
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor()
                ])
            # elif spatial_transform == "resize":
            #     self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        """
        json file
        {"caption": [
            {"scene_id1": scene_id1,
             "caption1": caption1},
            {"scene_id2": scene_id2,
             "caption2": caption2},
            ...
        ]}
        """
        self.metadata = []
        for subset in self.subsets:
            caption_path = os.path.join(self.data_dir, subset, self.caption_file_name)
            with open(caption_path, 'r') as f:
                captions = json.load(f)
            self.metadata += captions["captions"]
            print(f'{self.data_dir}/{subset} >>> {len(self.metadata)} data samples loaded.')

    def _load_scene_mappings(self):
        """
        Build a mapping from indices to (folder, filename) for each scene.
        """
        self.scene_mappings = {}
        for sample in tqdm(self.metadata):
            scene_id = sample['scene_id']
            subset = sample['subset']
            if scene_id not in self.scene_mappings:
                scene_path = os.path.join(self.data_dir, subset, scene_id)
                # Load partition.json
                partition_path = os.path.join(scene_path, 'partition.json')
                with open(partition_path, 'r') as f:
                    partition = json.load(f)
                train_indices = partition['train']
                test_indices = partition['test']
                # Build index_to_file mapping for this scene
                index_to_file = {}
                # For train indices
                for i, idx in enumerate(train_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('train', filename)
                # For test indices
                for i, idx in enumerate(test_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('test', filename)
                # Store mapping
                self.scene_mappings[scene_id] = index_to_file

    def _get_scene_path(self, sample):
        scene_path = os.path.join(self.data_dir, sample['subset'], sample['scene_id'])
        return scene_path

    def __getitem__(self, index):
        # try:
        index = index % len(self.metadata)
        sample = self.metadata[index]
        scene_id = sample['scene_id']
        scene_path = self._get_scene_path(sample)
        index_to_file = self.scene_mappings[scene_id]

        # Get train and test indices
        train_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'train']
        test_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'test']

        train_indices.sort()
        test_indices.sort()

        # Randomly pick stride between 1 and 5
        stride = get_valid_stride(train_indices, test_indices, self.video_length)
        selected_train_indices = sorted(train_indices[::stride])
        selected_test_indices = sorted(test_indices[::stride])
        
        selected_indices = selected_close_indices(selected_train_indices, selected_test_indices, 
                                                self.video_length, stride)

        if len(selected_indices) > self.video_length:
            selected_indices = selected_indices[:self.video_length]
        elif len(selected_indices) < self.video_length:
            repeats = (self.video_length + len(selected_indices) - 1) // len(selected_indices)
            selected_indices = (selected_indices * repeats)[:self.video_length]
        selected_indices.sort()

        crop_params = None
        # Load images and apply transformations
        images = []
        renders = []
        images_z = []
        renders_z = []
        
        guidances = []
        cond_frame_idx = []
        # for i, idx in enumerate(selected_indices):
        #     folder, filename = index_to_file[idx]
        #     if folder == 'train':
        #         cond_frame_idx.append(i)
        #     image_path = os.path.join(scene_path, folder, 'ours_30000/gt', filename)
        #     render_path = os.path.join(scene_path, folder, 'ours_30000/renders', filename)
        #     image = Image.open(image_path).convert('RGB')
        #     render = Image.open(render_path).convert('RGB')
        #     # if self.spatial_transform is not None:
        #     image = self.spatial_transform(image)
        #     render = self.spatial_transform(render)
        #     # load guidances
        #     images.append(image)
        #     renders.append(render)
            
        #     image_z_path = image_path.replace(".png", "_z.pt")
        #     image_z = torch.load(image_z_path)[0] # [4, H//8, W//8]
        #     images_z.append(image_z)
            
        #     render_z_path = render_path.replace(".png", "_z.pt")
        #     render_z = torch.load(render_z_path)[0] # [4, H//8, W//8]
        #     renders_z.append(render_z)
            
        # Stack images into a tensor [C, T, H, W]
        # frames = torch.stack(images, dim=1)
        # renders = torch.stack(renders, dim=1)
        # frames_z = torch.stack(images_z, dim=1)
        # renders_z = torch.stack(renders_z, dim=1)
        # cond_frame_idx = random.choice(cond_frame_idx)

        # Load extrinsics and select corresponding ones
        cam_extrinsics = np.load(os.path.join(scene_path, 'cam_extrinsics.npy'))

        if self.relative_pose:
            # (1) normalization 
            # get center and diag of the scene
            center = np.mean(cam_extrinsics[:, :3, 3], axis=0)
            diag = np.max(np.linalg.norm(cam_extrinsics[:, :3, 3] - center, axis=1))
            # get relative pose
            cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] - center
            cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] / diag
            selected_extrinsics = cam_extrinsics[selected_indices]
            
            # (2) get relative pose
            criterion = cam_extrinsics[selected_train_indices][0]
            criterion_inv = np.linalg.inv(criterion)
            selected_extrinsics = np.einsum('ij,kjl->kil', criterion_inv, selected_extrinsics)
        else:
            selected_extrinsics = cam_extrinsics[selected_indices]
        selected_extrinsics = selected_extrinsics[:, :3, :].reshape(-1, 12)
        selected_extrinsics = torch.tensor(selected_extrinsics)
        # Prepare data dictionary
        frames_embed = None
        
        frames_z = torch.load(os.path.join(scene_path, 'gt_z.pt'))
        renders_z = torch.load(os.path.join(scene_path, 'render_z.pt'))
        frames_embed = torch.load(os.path.join(scene_path, 'gt_embed.pt'), map_location='cpu')
        
        frames_z = frames_z[:, :, selected_indices, :, :][0]
        renders_z = renders_z[:, :, selected_indices, :, :][0]
        frames_embed = frames_embed[random.choice(selected_train_indices), ...]
        
        dino_feats_pca = torch.load(os.path.join(scene_path, 'dino_feats_pca.pth'))
        # dino_feats = torch.load(os.path.join(scene_path, 'dino_feats_all.pth'))
        control = dino_feats_pca[selected_indices]
        control = control.permute(0, 3, 1, 2) # (T, H, W, 3) -> (T, 3, H, W)
        
        frames = None
        renders = None
        
        data = {
            # 'video': frames,  # [C, T, H, W]
            'video_z': frames_z,  # [4, T, H//8, W//8]
            # 'renders': renders,  # [4, T, H, W]
            'renders_z': renders_z,  # [4 , T, H//8, W//8]
            'caption': sample['caption'],
            'camera_pose': selected_extrinsics,  # [T, ...]
            'path': scene_path,
            'fps': 5, # TODO : change it to use extrinsics
            'scene_id': scene_id,
            'cond_frame_idx': cond_frame_idx,
            "embed": frames_embed,
            "control": control
        }
        if self.guidance_type != 'none':
            data['guidance'] = guidances
        # iterate data and print shape if value is tensor
        # for key, value in data.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"[DEBUG] : {key} shape = {value.shape}")
        #     else:
        #         print(f"[DEBUG] : {key} = {value}")
        return data
        # except:
        #     print(f"[ERROR] : DL3DV {index} is not valid")
        #     return self.__getitem__(random.randint(0, len(self.metadata)-1))

    def __len__(self):
        return len(self.metadata)

class GSDataDL3DV_prepare(Dataset):
    """
    GSDataDL3DV Dataset.
    Assumes GSDataDL3DV data is structured as follows.
    GSDataDL3DV/
    """
    def __init__(self,
                 data_dir,
                 caption_file_name,
                 subsets,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 guidance_type='none',
                 relative_pose=False,
                 ):
        self.data_dir = data_dir
        self.subsets = subsets
        self.caption_file_name = caption_file_name
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.guidance_type = guidance_type
        self.relative_pose = relative_pose

        self._load_metadata()
        self._load_scene_mappings()

        spatial_transform = "resize_center_crop" # enforce
        # Define spatial transformations if any
        if spatial_transform is not None:
            if spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor()
                ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        """
        json file
        {"caption": [
            {"scene_id1": scene_id1,
             "caption1": caption1},
            {"scene_id2": scene_id2,
             "caption2": caption2},
            ...
        ]}
        """
        self.metadata = []
        for subset in self.subsets:
            caption_path = os.path.join(self.data_dir, subset, self.caption_file_name)
            with open(caption_path, 'r') as f:
                captions = json.load(f)
            self.metadata += captions["captions"]
            print(f'{self.data_dir}/{subset} >>> {len(self.metadata)} data samples loaded.')

    def _load_scene_mappings(self):
        """
        Build a mapping from indices to (folder, filename) for each scene.
        """
        self.scene_mappings = {}
        for sample in tqdm(self.metadata):
            scene_id = sample['scene_id']
            subset = sample['subset']
            if scene_id not in self.scene_mappings:
                scene_path = os.path.join(self.data_dir, subset, scene_id)
                # Load partition.json
                partition_path = os.path.join(scene_path, 'partition.json')
                with open(partition_path, 'r') as f:
                    partition = json.load(f)
                train_indices = partition['train']
                test_indices = partition['test']
                # Build index_to_file mapping for this scene
                index_to_file = {}
                # For train indices
                for i, idx in enumerate(train_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('train', filename)
                # For test indices
                for i, idx in enumerate(test_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('test', filename)
                # Store mapping
                self.scene_mappings[scene_id] = index_to_file

    def _get_scene_path(self, sample):
        scene_path = os.path.join(self.data_dir, sample['subset'], sample['scene_id'])
        return scene_path

    def __getitem__(self, index):
        # try:
        index = index % len(self.metadata)
        sample = self.metadata[index]
        scene_id = sample['scene_id']
        scene_path = self._get_scene_path(sample)
        index_to_file = self.scene_mappings[scene_id]

        # Get train and test indices
        train_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'train']
        test_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'test']

        train_indices.sort()
        test_indices.sort()

        selected_indices = train_indices + test_indices

        crop_params = None
        # Load images and apply transformations
        images = []
        image_paths = []
        renders = []
        render_paths = []
        for i, idx in enumerate(selected_indices):
            folder, filename = index_to_file[idx]
            image_path = os.path.join(scene_path, folder, 'ours_30000/gt', filename)
            render_path = os.path.join(scene_path, folder, 'ours_30000/renders', filename)
            image = Image.open(image_path).convert('RGB')
            render = Image.open(render_path).convert('RGB')
            # if self.spatial_transform is not None:
            image = self.spatial_transform(image)
            render = self.spatial_transform(render)
            images.append(image)
            renders.append(render)
            image_paths.append(image_path)
            render_paths.append(render_path)
            
        # Stack images into a tensor [C, T, H, W]
        frames = torch.stack(images, dim=1)
        renders = torch.stack(renders, dim=1)
        
        # Prepare data dictionary
        data = {
            'video': frames,  # [C, T, H, W]
            'renders': renders,  # [C, T, H, W]
            'image_paths': image_paths,
            'render_paths': render_paths,
        }
        
        return data
        # except:
        #     print(f"[ERROR] : DL3DV {index} is not valid")
        #     return self.__getitem__(random.randint(0, len(self.metadata)-1))

    def __len__(self):
        return len(self.metadata)

class GSData360(Dataset):
    """
    GSData360 Dataset.
    Assumes GSData360 data is structured as follows.
    GSData360/
    """
    def __init__(self,
                 data_dir,
                 caption_file_name,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 guidance_type='none',
                 relative_pose=False
                 ):
        self.data_dir = data_dir
        self.caption_file_name = caption_file_name
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.guidance_type = guidance_type
        self.relative_pose = relative_pose

        self._load_metadata()
        self._load_scene_mappings()

        # Define spatial transformations if any
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                # self.spatial_transform = transforms.RandomCrop(self.resolution)
                self.spatial_transform = "random_crop"
            elif spatial_transform == "resize_random_crop":
                self.spatial_transform = "resize_random_crop"
            # elif spatial_transform == "center_crop":
            #     self.spatial_transform = transforms.Compose([
            #         transforms.CenterCrop(resolution),
            #     ])            
            # elif spatial_transform == "resize_center_crop":
            #     self.spatial_transform = transforms.Compose([
            #         transforms.Resize(min(self.resolution)),
            #         transforms.CenterCrop(self.resolution),
            #     ])
            # elif spatial_transform == "resize":
            #     self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        """
        json file
        {"caption": [
            {"scene_id1": scene_id1,
             "caption1": caption1},
            {"scene_id2": scene_id2,
             "caption2": caption2},
            ...
        ]}
        """
        self.metadata = []
        caption_path = os.path.join(self.data_dir, self.caption_file_name)
        with open(caption_path, 'r') as f:
            captions = json.load(f)
        self.metadata += captions["captions"]
        print(f'{self.data_dir} >>> {len(self.metadata)} data samples loaded.')

    def _load_scene_mappings(self):
        """
        Build a mapping from indices to (folder, filename) for each scene.
        """
        self.scene_mappings = {}
        for sample in tqdm(self.metadata):
            scene_id = sample['scene_id']
            subset = ""
            if scene_id not in self.scene_mappings:
                scene_path = os.path.join(self.data_dir, subset, scene_id)
                # Load partition.json
                partition_path = os.path.join(scene_path, 'partition.json')
                with open(partition_path, 'r') as f:
                    partition = json.load(f)
                train_indices = partition['train']
                test_indices = partition['test']
                # Build index_to_file mapping for this scene
                index_to_file = {}
                # For train indices
                for i, idx in enumerate(train_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('train', filename)
                # For test indices
                for i, idx in enumerate(test_indices):
                    filename = f'{i:05d}.png'
                    index_to_file[idx] = ('test', filename)
                # Store mapping
                self.scene_mappings[scene_id] = index_to_file

    def _get_scene_path(self, sample):
        scene_path = os.path.join(self.data_dir, sample['scene_id'])
        return scene_path

    def __getitem__(self, index):
        try:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            scene_id = sample['scene_id']
            scene_path = self._get_scene_path(sample)
            index_to_file = self.scene_mappings[scene_id]

            # Get train and test indices
            train_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'train']
            test_indices = [idx for idx in index_to_file.keys() if index_to_file[idx][0] == 'test']

            train_indices.sort()
            test_indices.sort()

            # Randomly pick stride between 1 and 5
            stride = get_valid_stride(train_indices, test_indices, self.video_length)
            selected_train_indices = sorted(train_indices[::stride])
            selected_test_indices = sorted(test_indices[::stride])
            
            selected_indices = selected_close_indices(selected_train_indices, selected_test_indices, 
                                                    self.video_length, stride)

            if len(selected_indices) > self.video_length:
                selected_indices = selected_indices[:self.video_length]
            elif len(selected_indices) < self.video_length:
                repeats = (self.video_length + len(selected_indices) - 1) // len(selected_indices)
                selected_indices = (selected_indices * repeats)[:self.video_length]
            selected_indices.sort()

            crop_params = None
            # Load images and apply transformations
            images = []
            renders = []
            guidances = []
            cond_frame_idx = []
            for i, idx in enumerate(selected_indices):
                folder, filename = index_to_file[idx]
                if folder == 'train':
                    cond_frame_idx.append(i)
                image_path = os.path.join(scene_path, folder, 'ours_30000/gt', filename)
                render_path = os.path.join(scene_path, folder, 'ours_30000/renders', filename)
                image = Image.open(image_path).convert('RGB')
                render = Image.open(render_path).convert('RGB')
                
                if self.guidance_type != 'none':
                    # guidance_file_name = f"frame_{idx+1:05d}.png"
                    guidance_file_name = filename
                    if self.guidance_type == 'visibility_mask_global_2ndpick_th0.5':
                        guidance_path = os.path.join(scene_path, folder, f'ours_30000/{self.guidance_type}', guidance_file_name)
                        guidance = Image.open(guidance_path).convert('L') # grayscale
                        # print("[DEBUG] : guidance.size = ", guidance.size)
                        # guidances.append(transforms.ToTensor()(guidance))
                    else:
                        raise NotImplementedError(f"[ERROR] : Guidance type not supported {self.guidance_type}")
                else:
                    guidance = None
                
                if self.spatial_transform == "resize_random_crop":
                    image = transforms.Resize(max(self.resolution))(image)
                    render = transforms.Resize(max(self.resolution))(render)
                    if guidance is not None:
                        guidance = transforms.Resize(max(self.resolution))(guidance)
                
                if crop_params is None:
                    if self.spatial_transform == "random_crop" or self.spatial_transform == "resize_random_crop":
                        dummy = torch.zeros(3, image.size[1], image.size[0]) # get resolution from image (PIL)
                        crop_params = transforms.RandomCrop.get_params(dummy, self.resolution)
                    
                if self.spatial_transform == "random_crop" or self.spatial_transform == "resize_random_crop":
                    # Sample crop params from the first image
                    a,b,c,d = crop_params
                    image = F.crop(image, a, b, c, d)
                    render = F.crop(render, a, b, c, d)
                    # Apply to guidance
                    if guidance is not None:
                        guidance = F.crop(guidance, a, b, c, d)
                    image = transforms.ToTensor()(image)
                    render = transforms.ToTensor()(render)
                    if guidance is not None:
                        guidance = transforms.ToTensor()(guidance)
                images.append(image)
                renders.append(render)
                if guidance is not None:
                    guidances.append(guidance)
                
            # Stack images into a tensor [C, T, H, W]
            frames = torch.stack(images, dim=1)
            renders = torch.stack(renders, dim=1)
            cond_frame_idx = random.choice(cond_frame_idx)
            if self.guidance_type != 'none':
                guidances = torch.stack(guidances, dim=1)

            # Load extrinsics and select corresponding ones
            cam_extrinsics = np.load(os.path.join(scene_path, 'cam_extrinsics.npy'))
            if self.relative_pose:
                # (1) normalization 
                # get center and diag of the scene
                center = np.mean(cam_extrinsics[:, :3, 3], axis=0)
                diag = np.max(np.linalg.norm(cam_extrinsics[:, :3, 3] - center, axis=1))
                # get relative pose
                cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] - center
                cam_extrinsics[:, :3, 3] = cam_extrinsics[:, :3, 3] / diag
                selected_extrinsics = cam_extrinsics[selected_indices]
                
                # (2) get relative pose
                criterion = cam_extrinsics[selected_train_indices][0]
                criterion_inv = np.linalg.inv(criterion)
                selected_extrinsics = np.einsum('ij,kjl->kil', criterion_inv, selected_extrinsics)
            else:
                selected_extrinsics = cam_extrinsics[selected_indices]
            selected_extrinsics = selected_extrinsics[:, :3, :].reshape(-1, 12)
            selected_extrinsics = torch.tensor(selected_extrinsics)
            # Prepare data dictionary
            data = {
                'video': frames,  # [C, T, H, W]
                'renders': renders,  # [C, T, H, W]
                'caption': sample['caption'],
                'camera_pose': selected_extrinsics,  # [T, ...]
                'path': scene_path,
                'fps': 5, # TODO : change it to use extrinsics
                'scene_id': scene_id,
                'cond_frame_idx': cond_frame_idx
            }
            if self.guidance_type != 'none':
                data['guidance'] = guidances
            # for key, value in data.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"[DEBUG] : {key} shape = {value.shape}")
            #     else:
            #         print(f"[DEBUG] : {key} = {value}")
            return data
        except:
            return self.__getitem__(random.randint(0, len(self.metadata)-1))

    def __len__(self):
        return len(self.metadata)

class GSDataConcat(Dataset):
    def __init__(self,
                 _360_data_dir="",
                 _360_caption_file_name="",
                 dl3dv_data_dir="",
                 dl3dv_caption_file_name="",
                 re10k_data_dir="",
                 re10k_caption_file_name="",
                 subsets=None,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 guidance_type='none',
                 relative_pose=False):
        self.datasets = []
        # Simplified instantiation
        self.dl3dv = self._create_dl3dv_dataset(dl3dv_data_dir, dl3dv_caption_file_name,
                                                subsets, subsample, video_length, resolution, frame_stride, 
                                                frame_stride_min, spatial_transform, crop_resolution, fps_max, 
                                                load_raw_resolution, fixed_fps, random_fs, guidance_type, relative_pose)
        if self.dl3dv is not None:
            self.datasets.append(self.dl3dv)
        
        self._360data = self._create_360_dataset(_360_data_dir, _360_caption_file_name,
                                                subsample, video_length, resolution, frame_stride, 
                                                frame_stride_min, spatial_transform, crop_resolution, fps_max, 
                                                load_raw_resolution, fixed_fps, random_fs, guidance_type, relative_pose)
        if self._360data is not None:
            self.datasets.append(self._360data)
        
        self.re10kdata = self._create_re10k_dataset(re10k_data_dir, re10k_caption_file_name, None,
                                                subsample, video_length, resolution, frame_stride, 
                                                frame_stride_min, spatial_transform, crop_resolution, fps_max, 
                                                load_raw_resolution, fixed_fps, random_fs, guidance_type, relative_pose, True) # skip_subset = True
        if self.re10kdata is not None:
            self.datasets.append(self.re10kdata)
        
        # ConcatDataset
        self.dataset = ConcatDataset(self.datasets)

    def _create_dl3dv_dataset(self, data_dir, caption_file_name, *args):
        print(f"Creating DL3DV dataset with args: {args}")
        if data_dir == "":
            return None
        else:
            return GSDataDL3DV(data_dir, caption_file_name, *args)

    def _create_360_dataset(self, data_dir, caption_file_name, *args):
        print(f"Creating 360 dataset with args: {args}")
        if data_dir == "":
            return None
        else:
            return GSData360(data_dir, caption_file_name, *args)

    def _create_re10k_dataset(self, data_dir, caption_file_name, *args):
        print(f"Creating RE10K dataset with args: {args}")
        if data_dir == "":
            return None
        else:
            return GSDataDL3DV(data_dir, caption_file_name, *args)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

if __name__== "__main__":
    meta_path = "" ## path to the meta file
    data_dir = "" ## path to the data directory
    save_dir = "" ## path to the save directory
    dataset = WebVid(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256,448],
                 frame_stride=4,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)

