# %%
import functools
from dust3r.demo import get_reconstructed_scene
from torch.nn.functional import mse_loss
import numpy as np

# %%
import os
import torch
import tempfile

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, main_demo, set_print_with_timestamp
from dust3r.inference import inference,inference_teacher
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
import matplotlib.pyplot as pl

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
device="cuda"
batch_size=64
model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"
weights="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if weights is not None:
    weights_path = weights
else:
    weights_path = "naver/" + model_name
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)


# %%
import glob
# Load all images
# gt_depths= load_images(list(glob.glob("/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/depth/*.png"))[:5],grayscale=True, size=512)
# imgs = load_images(list(glob.glob("/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/rgb/*.png"))[:5], size=512)

# base_path="/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/depth"
# gt_depths=[
#     base_path+"/1305031102.160407.png",
#     base_path+"/1305031102.194330.png",
#     base_path+"/1305031102.226738.png",
#     base_path+"/1305031102.262886.png",
#     base_path+"/1305031102.295279.png",
# ]
# # gt_depths= load_images(gt_depths,grayscale=True, size=512)
# base_path="/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/rgb"
# imgs=[
#     base_path+"/1305031102.175304.png",
#     base_path+"/1305031102.211214.png",
#     base_path+"/1305031102.243211.png",
#     base_path+"/1305031102.275326.png",
#     base_path+"/1305031102.311267.png",
# ]
gt_depths= list(glob.glob("data/depth/*.png"))
imgs = list(glob.glob("data/rgb/*.png"))

def create_iterator(gt_depths,imgs,batch_size=32):
    m=0
    while (m+1)*batch_size<len(imgs):
        end=min(m+1*batch_size,len(imgs))
        start=max(0,m*batch_size)
        yield load_images(gt_depths[start:end],grayscale=True,size=512),load_images(imgs[start:end],size=512)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from itertools import repeat
import collections.abc


# %%
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


# %%
# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

# %%
class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) 


# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from itertools import repeat
import collections.abc

class Dust3RStudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = torchvision.models.vit_b_16(pretrained=True)
        # self.preprocessing = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
        # for param in model.parameters():
        #         param.requires_grad = True
        # # Remove classifier head
        # self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
        # Feature extractor (encoder)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), stride=(3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (2, 2), stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(512, 768, (2, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))  # This will resize output to exactly 32x32
        )
                
        # # Downscale to reduce feature dimensions
        self.downscale = nn.Sequential(
            nn.Conv1d(768, 512, 3,stride=4),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3,stride=4),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3,stride=4),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3,stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )
        
        # # Reconstruction layer combines features
        # self.reconstruction = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        # )
        
        # # Decoder for 3D reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
        )
        

        self.point_cloud_head = nn.Linear(4096, 3 * 1024)  # xyz coordinates for 1024 points
        self.depth_head = nn.Linear(4096, 384 * 512)  # depth map of size 128x128
        
    def forward(self, img1, img2):
        
        # vit = self.modelhts=ViT_B_16_Weights.DEFAULT)
        # img1 = self.preprocessing(img1)
        # img2 = self.preprocessing(img2)
        
        feat1 = self.feature_extractor(torch.cat([img1,img2],dim=0))
        feat2 = self.feature_extractor(torch.cat([img2,img1],dim=0))

        feat1 = torch.flatten(feat1,start_dim=2)
        feat2 = torch.flatten(feat2,start_dim=2)
        # print(feat1.shape)
        

        img1 = self.downscale(feat1)
        img2 = self.downscale(feat2)



        # Combine features in both directions
        # print(img1.shape,img2.shape,torch.flatten(feat1, start_dim=1))
        # print(feat1.shape,feat2.shape,torch.flatten(feat1, start_dim=1))
        # combined_feat1 = torch.cat([feat1, feat2], dim=-1)
        # combined_feat2 = torch.cat([feat2, feat1], dim=-1)
        
        # Initial reconstruction
        # recon_feat1 = self.reconstruction(img1)
        # recon_feat2 = self.reconstruction(img2)
        
        # # Decode features to get final output
        decoded1 = self.decoder(img1)
        decoded2 = self.decoder(img2)
        
        # # Generate different output types
        point_cloud1 = self.point_cloud_head(decoded1).reshape(-1, 1024, 3)
        point_cloud2 = self.point_cloud_head(decoded2).reshape(-1, 1024, 3)
        
        depth_map1 = self.depth_head(decoded1).reshape(-1, 384, 512)
        depth_map2 = self.depth_head(decoded2).reshape(-1, 384, 512)
        
        return {
            'features': [feat1, feat2],
            # 'reconstruction': [recon_feat1, recon_feat2],
            'point_clouds': [point_cloud1, point_cloud2],
            'depth_maps': [depth_map1, depth_map2]
        }
# model=Dust3RStudentModel()
# model(torch.randn((2,3,512,512)),torch.randn((2,3,512,512)))

# %%

def create_dust3r_student(pretrained_teacher=None):
    """
    Create a student model, optionally initializing from a teacher model.
    """
    model = Dust3RStudentModel()
    
    if pretrained_teacher is not None:
        # Initialize from teacher (separate function)
        model = initialize_from_teacher(model, pretrained_teacher)
    
    return model


def initialize_from_teacher(student, teacher):
    """
    Initialize student model from teacher using knowledge distillation techniques.
    This function handles the mismatch in architecture sizes.
    """
    # Implementation depends on specific teacher architecture
    # This is a placeholder for the actual implementation
    return student


# %%


def knowledge_distillation_loss(student_outputs, teacher_outputs, gt_data, alpha=0.5, temperature=2.0):
    student_features = student_outputs['features']
    
    # print(student_features[0].shape,teacher_outputs['feat_1'].shape,teacher_outputs['feat_2'].shape,gt_data)
    cossim=nn.CosineSimilarity()

    teacher_features = [teacher_outputs['feat_1'],teacher_outputs['feat_2']]
    # print(teacher_features[0].shape,student_features[0].shape)
    distillation_loss = 0
    for sf, tf in zip(student_features, teacher_features):
        # Normalize features
        sf = F.normalize(sf, dim=1)
        tf = F.normalize(tf, dim=1)
        sim = cossim(sf,tf)
        # Compute soft targets
        soft_targets = F.softmax(sim / temperature, dim=1).cuda()
        # Compute cross entropy
        distillation_loss += F.cross_entropy(soft_targets, torch.arange(soft_targets.size(0)).cuda())


    student_reconstruction = student_outputs['depth_maps']
    # student_points = torch.cat([student_reconstruction[0], student_reconstruction[0]], dim=1)
    # student_points = student_reconstruction[0]
    # Find way to deal with variable shape
    # print(student_points.shape,gt_data.shape)
    reconstruction_loss = F.mse_loss(student_reconstruction[0], (gt_data))+F.mse_loss(student_reconstruction[1], gt_data)
    alpha=torch.Tensor(alpha).cuda()
    # Combined loss
    total_loss = (alpha * reconstruction_loss + (1 - alpha) * distillation_loss).mean()
    # print
    return total_loss

# %%
def train_dust3r_student(student, teacher,optimizer, epochs=10):
    """
    Train the student model using knowledge distillation
    """
    teacher.eval()  # Teacher model in evaluation mode
    student.train()  # Student model in training mode
    train_loader=create_iterator(gt_depths,imgs,batch_size=batch_size)
    for epoch in range(epochs):
        for batch in train_loader:
            # Get data
            gt_data,img = batch
            pairs = make_pairs(img, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference_teacher(pairs, model, device, batch_size=batch_size)
            # print(list(map(lambda x:x[0]["img"],pairs)))
            # print(output["pred1"]['pts3d'])
            # Forward pass through student
            # print("Feon", output['feat_1'].shape,output['pred1']["pts3d"].shape, output['feat_2'].shape)
            student_outputs = student(*list(map(lambda x:x[0]["img"],pairs)))
            # print(output["pred1"].shape)
            # Compute loss
            loss = knowledge_distillation_loss(student_outputs, output, *list(map(lambda x:x["img"],gt_data)))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return student

# %%

student_model = create_dust3r_student(pretrained_teacher=model)

# %%
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# Load your dataset (with paired images)
# train_loader = get_dust3r_dataset()  # Your data loading function

# Train the student using knowledge distillation
trained_student = train_dust3r_student(
    student=student_model,
    teacher=model,
    optimizer=optimizer,
    epochs=10
)

# Save the trained model
# torch.save(trained_student.state_dict(), "dust3r_student.pth")

# %%
# # recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
# silent=False
# schedule="cosine"
# niter=300
# # 

# pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
# output = inference(pairs, model, device, batch_size=batch_size)

# mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
# scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
# lr = 0.01

# if mode == GlobalAlignerMode.PointCloudOptimizer:
#     loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

# # also return rgb, depth and confidence imgs
# # depth is normalized with the max value for all images
# # we apply the jet colormap on the confidence maps
# rgbimg = scene.imgs
# depths = to_numpy(scene.get_depthmaps())
# confs = to_numpy([c for c in scene.im_conf])
# cmap = pl.get_cmap('jet')
# depths_max = max([d.max() for d in depths])
# depths = [d / depths_max for d in depths]
# confs_max = max([d.max() for d in confs])
# confs = [cmap(d / confs_max) for d in confs]

# imgs = []
# depth_rgb=[]
# for i in range(len(rgbimg)):
#     imgs.append(rgbimg[i])
#     imgs.append(rgb(depths[i]))
#     depth_rgb.append(depths[i])
#     imgs.append(rgb(confs[i]))


# %%
# gt_depths=torch.stack(list(map(lambda x:x["img"],gt_depths))).squeeze()

# %%
# depth_rgb=torch.from_numpy(np.stack(depth_rgb))


# %%
# import torchvision

# output=torch.concat([depth_rgb,gt_depths],dim=1).reshape((768*5,512))
# torchvision.utils.save_image(output,"output.png")



# %%
# # https://arxiv.org/pdf/2312.14132
# abs_rl=torch.abs(gt_depths - depth_rgb)/depth_rgb

# %%
# abs_rl


