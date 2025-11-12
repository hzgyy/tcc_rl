# import os
import numpy as np

# import utils

from .train_tcc import AlignNet
import torch

# import random
# import argparse
# import glob
# from natsort import natsorted
# from tqdm import tqdm
# import cv2
# from PIL import Image, ImageDraw, ImageFont

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from . import transforms_video as tv
# from config import CONFIG

class ProgressEstimator():
    def __init__(self,ckpt,demonstrations,demonstation_labels,device,k=3):
        # device = f"cuda:{args.device}"
        self.device = device
        self.model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
        self.model.to(device)
        self.model.eval()
        # grad off
        torch.set_grad_enabled(False)
        self.transform = tv.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.demo_embs = self.get_embs(demonstrations)  #np.array (N,128)
        self.demo_labels = demonstation_labels          #np.array (N,)
        self.k = k

    def get_embs(self,imgs):
        # turn imgs into correct tensor
        if type(imgs) == np.ndarray:
            imgs = tv.ToTensorVideo(imgs)
        imgs = self.transform(imgs)
        
        # get embeddings
        a_X = imgs.to(self.device).unsqueeze(0)
        original = a_X.shape[1]//2
        a_emb = self.model(a_X)
        #print(f"emb shape:{a_emb.shape},original:{original}")
        a_emb = a_emb[:, :original,:]

        a_emb_reduced = a_emb.squeeze(0).detach().cpu().numpy()
        return a_emb_reduced
    
    def get_estimate(self,imgs):
        query_embs = self.get_embs(imgs)    #(M,128)
        # I assume all the embs are normalized
        sim = query_embs @ self.demo_embs.T   # (M,N)
        knn_indices = np.argpartition(-sim, self.k, axis=1)[:, :self.k]   # (M,K)
        estimated = np.mean(self.demo_labels[knn_indices], axis=1)    # (M,)
        return estimated

def main():
    device = f"cuda:{0}"
    ckpt_path = "/media/mani/Data/gyy_workspace/IBRL/ibrl/lav/trained_model/final_model_l2norm-True_sigma-10_alpha-0.5_lr-0.0001_bs-4.pth"
    demonstrations = torch.randn(10,3,224,224)
    demonstration_labels = np.arange(10)/10
    pro_est = ProgressEstimator(ckpt_path,demonstrations,demonstration_labels,device)
    imgs = torch.randn(6,3,224,224)
    est = pro_est.get_estimate(imgs)
    print(est)

if __name__ == '__main__':
    main()