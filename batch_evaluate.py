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
from dust3r.inference import inference
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
batch_size=2
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
gt_depths= list(glob.glob("data/depth/*.png"))
imgs = list(glob.glob("data/rgb/*.png"))
def create_iterator(imgs,gt_depths,batch=32):
	m=0
	while  (m+1)*batch < len(imgs):
		end=min(m+1,len(imgs))*batch
		start=m*batch
		yield imgs[start:end],gt_depths[start:end]
		m+=1




#
#base_path="/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/depth"

#gt_depths=[
   # base_path+"/1305031102.160407.png",
   # base_path+"/1305031102.194330.png",
  #  base_path+"/1305031102.226738.png",
 #   base_path+"/1305031102.262886.png",
#    base_path+"/1305031102.295279.png",
#]
#gt_depths= load_images(gt_depths,grayscale=True, size=512)
#base_path="/home/kojogyaase/Projects/Research/3D_Recon/dependencies/dust3r/data/rgbd_dataset_freiburg1_xyz/rgb"

#imgs=[
 #   base_path+"/1305031102.175304.png",
  #  base_path+"/1305031102.211214.png",
   # base_path+"/1305031102.243211.png",
   # base_path+"/1305031102.275326.png",
    #base_path+"/1305031102.311267.png",
#]
#imgs = load_images(imgs, size=512)


# %%
# recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
silent=False
schedule="cosine"
niter=300
# 
data = create_iterator(imgs,gt_depths)
errors=[]
for  imgs,gt_depths in data:
	imgs=load_images(imgs,size=512)
	gt_depths=load_images(gt_depths,grayscale=True,size=512)
	pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
	#print(pairs)
	output = inference(pairs, model, device, batch_size=batch_size)

	mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
	scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

	lr = 0.01

	if mode == GlobalAlignerMode.PointCloudOptimizer:
    		loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

	# also return rgb, depth and confidence imgs
	# depth is normalized with the max value for all images
	# we apply the jet colormap on the confidence maps
	rgbimg = scene.imgs
	depths = to_numpy(scene.get_depthmaps())
	confs = to_numpy([c for c in scene.im_conf])
	cmap = pl.get_cmap('jet')
	depths_max = max([d.max() for d in depths])
	depths = [d / depths_max for d in depths]
	confs_max = max([d.max() for d in confs])
	confs = [cmap(d / confs_max) for d in confs]

	imgs = []
	depth_rgb=[]
	for i in range(len(rgbimg)):
    		imgs.append(rgbimg[i])
    		imgs.append(rgb(depths[i]))
    		depth_rgb.append(depths[i])
    		imgs.append(rgb(confs[i]))


	# %%
	gt_depths=torch.stack(list(map(lambda x:x["img"],gt_depths))).squeeze()

	# %%
	depth_rgb=torch.from_numpy(np.stack(depth_rgb))


	# %%
	import torchvision

	output=torch.concat([depth_rgb,gt_depths],dim=1).reshape((768*32,512))
	torchvision.utils.save_image(output,"output.png")
	# %%
	# https://arxiv.org/pdf/2312.14132
	abs_rl=torch.mean(torch.abs(gt_depths - depth_rgb)/depth_rgb)
	print("ABS RL",abs_rl)
	errors.append(abs_rl)

print("Mean Abs REL",torch.mean(torch.stack(errors)))


