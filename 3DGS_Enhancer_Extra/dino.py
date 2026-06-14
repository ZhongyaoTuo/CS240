import torch
import json
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

model = dinov2_vitl14.eval()
model.to('cuda')

# extract DINO features
# extract DINO PCA features
# save all

if __name__ == "__main__":
    
    scene_path = "/workspace/dataset/examples/1K/62f3a1d6da8e55251a59d9e8f0b0195efa1251275ffb0f34a1f16146fe277908_0.7"

    with open(os.path.join(scene_path, "partition.json"), "r") as f:
        partition = json.load(f)

    train_indices = sorted(partition["train"])
    test_indices = sorted(partition["test"])

    train_img_gt_dir = os.path.join(scene_path, "train/ours_30000/gt")
    test_img_gt_dir = os.path.join(scene_path, "test/ours_30000/gt")
    
    train_img_render_dir = os.path.join(scene_path, "train/ours_30000/renders")
    test_img_render_dir = os.path.join(scene_path, "test/ours_30000/renders")
    
    train_img_paths = [os.path.join(train_img_gt_dir, f"{i:05d}.png") for i in range(len(train_indices))]
    test_img_paths = [os.path.join(test_img_gt_dir, f"{i:05d}.png") for i in range(len(test_indices))]
    
    train_render_paths = [os.path.join(train_img_render_dir, f"{i:05d}.png") for i in range(len(train_indices))]
    test_render_paths = [os.path.join(test_img_render_dir, f"{i:05d}.png") for i in range(len(test_indices))]
    
    # reorder train_imgs, train_renders, test_imgs, test_renders using indices
    
    n_total = max(max(train_indices), max(test_indices)) + 1
    
    imgs = []
        
    dummy = Image.open(train_img_paths[0])
    # h, w
    w, h = dummy.size
    resize_factor = 1
    
    resize_w = int(w // 14 * 14 * resize_factor)
    resize_h = int(h // 14 * 14 * resize_factor)

    imgs = torch.zeros((n_total, 3, resize_h, resize_w))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_h, resize_w)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_imgs = torch.zeros((len(train_indices), 3, resize_h, resize_w))
    for i, (idx, path) in enumerate(zip(train_indices, train_img_paths)):
        img = Image.open(path)
        
        # to Tensor
        img = transform(img)
        # print("train :", img.shape, idx)    
        train_imgs[i] = img
        
    test_imgs = torch.zeros((len(test_indices), 3, resize_h, resize_w))
    for i, (idx, path) in enumerate(zip(test_indices, test_img_paths)):
        img = Image.open(path)
        
        # to Tensor
        img = transform(img)
        # print("test :", img.shape, idx)    
        test_imgs[i] = img
        
    # print(imgs.shape)
    
    chunk = 8
    dino_feats_train = []
    dino_feats_test = []
    
    with torch.no_grad():
        for i in range(0, len(train_indices), chunk):
            chunk_imgs = train_imgs[i:i+chunk]
            chunk_feats = model.forward_features(chunk_imgs.cuda())
            patch_tokens = chunk_feats['x_norm_patchtokens']
            dino_feats_train.append(patch_tokens.cpu())
        
        for i in range(0, len(test_indices), chunk):
            chunk_imgs = test_imgs[i:i+chunk]
            chunk_feats = model.forward_features(chunk_imgs.cuda())
            patch_tokens = chunk_feats['x_norm_patchtokens']
            dino_feats_test.append(patch_tokens.cpu())
        
    dino_feats_train = torch.cat(dino_feats_train, dim=0)
    dino_feats_test = torch.cat(dino_feats_test, dim=0)
    
    dino_feats_all = torch.zeros((n_total, (resize_h // 14) * (resize_w // 14), 1024))
    dino_feats_all[train_indices] = dino_feats_train
    dino_feats_all[test_indices] = dino_feats_test
    
    torch.save(dino_feats_all, os.path.join(scene_path, "dino_feats_all.pth"))
    
    # PCA feature
    pca = PCA(n_components=3)
    pca.fit(dino_feats_all[train_indices[0]].reshape(-1, 1024).cpu().numpy())
    dino_feats_pca = torch.zeros((n_total, resize_h // 14, resize_w // 14, 3))
    for idx in train_indices:
        dino_feat_pca = pca.transform(dino_feats_all[idx].reshape(-1, 1024).cpu().numpy())
        dino_feats_pca[idx] = torch.from_numpy(dino_feat_pca).reshape((resize_h // 14), (resize_w // 14), 3)
    
    for idx in test_indices:
        dino_feat_pca = pca.transform(dino_feats_all[idx].reshape(-1, 1024).cpu().numpy())
        dino_feats_pca[idx] = torch.from_numpy(dino_feat_pca).reshape((resize_h // 14), (resize_w // 14), 3)
    
    torch.save(dino_feats_pca, os.path.join(scene_path, "dino_feats_pca.pth"))
    
    
    
    
    
        
        
    
