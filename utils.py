import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
from pytorchvideo.transforms import CutMix
from torchvision import transforms
import random
from einops import rearrange
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import scipy.stats

def InfoNCE_loss(m, n):

    btch, ch_m, d_m, h_m, w_m = m.size()
    btch, ch_n, d_n, h_n, w_n = n.size()
    # m = m.permute(0, 4, 2, 3, 1)
    m = m.contiguous().view(btch, -1)
    n = n.contiguous().view(btch, -1)
    # m = m.permute(1, 0, 2, 3, 4)
    # m = m.contiguous().view(-1, ch_m)
    # n = n.contiguous().view(ch_n, -1)
    similarity_scores = torch.matmul(m, torch.t(n))  # 矩阵乘法计算相似度得分

    # 计算相似度得分的温度参数
    temperature = 0.07

    # 计算logits
    logits = similarity_scores / temperature

    # 构建labels（假设有N个样本）
    N = n.size(0)
    labels = torch.arange(N).to(logits.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)

    return loss


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(4, 3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(4, 3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv3d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv3d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv3d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv3d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    return 10 * math.log10(1 / mse)


def psnrv2(mse, peak):
    # note: peak = max(I) where I ranged from 0 to 2 (considering mse is calculated when I is ranged -1 to 1)
    return 10 * math.log10(peak * peak / mse)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):
    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[
                                                     0] + 1) / 2)  # +1/2 probably the g() function. Normalize from 0-1. Although not exactly min() and max() value.
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins

def shuffle_index(x_len=8):
    idx = torch.randperm(x_len)
    s = torch.abs(idx[1:]- idx[:-1])
    while any(s == 1):
        idx = torch.randperm(x_len)
        s = torch.abs(idx[1:] - idx[:-1])
    return idx

def genAppAnoSmps(x):
    cm = CutMix(alpha=1.0, label_smoothing=0, num_classes=2)
    b = x.shape[0]
    lb = torch.zeros(b)
    return cm(x, lb)[0]

def cal_mot_smooth_lb(x, tmp_coef=10):
    if tmp_coef is None:
        tmp_coef = len(x)
    # x is index
    x_dif = torch.diff(x)
    # reverse
    rev_n = torch.sum(x_dif < 0)
    # skip
    skips = torch.sum(torch.abs(x_dif))
    # print(skips)

    abn_n = rev_n+skips-len(x)+1
    # print(abn_n)
    abn_n = 1 - torch.exp(-abn_n/tmp_coef)
    return abn_n

def genMotionAnoSmps(x):
    x_shuffle = x[:, :, shuffle_index(x.size()[2]), :, :]
    return x_shuffle



class ForegroundEstimate():
    def __init__(self, pth='E:/dataset/UCSD/UCSDped2/Train/', imgType='*.tif', imgSz=(240, 360)) -> None:
        super().__init__()
        self.pth = pth
        self.imgType = imgType
        self.imgsz = imgSz
        self.bg_crt = cv2.createBackgroundSubtractorMOG2()

    def estFg(self):
        fg_mask = np.zeros(self.imgsz)

        flders = Path(self.pth).glob("*")
        kernel = np.ones((2, 2), np.uint8)
        for fld in flders:
            if fld.is_dir():
                imgs = Path(fld).glob(self.imgType)
                for imn in imgs:
                    im = cv2.imread(str(imn))
                    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    # plt.imshow(im)
                    fg = self.bg_crt.apply(im)
                    fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY)[1]
                    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

                    # imgCnt+=1
                    fg = fg > 0
                    fg_mask = fg_mask + fg
        return fg_mask

    def estBinaryFg(self, fg_mask):
        # fg_mask = self.estFg()
        fg = fg_mask > np.mean(fg_mask)
        # plt.imshow(fg)
        return fg

    def estFgROI(self, fg_mask):
        # fg_mask = self.estFg()
        fg = fg_mask > np.mean(fg_mask)
        x, y, w, h = cv2.boundingRect(np.uint8(fg * 255))
        return x, y, w, h

class FgCutmix(torch.nn.Module):
        def __init__(self, imgs_mem_pth, prob_map_pth, p_thr=None, area_thr=None,
                     roi_width_range=(10, 25), roi_height_range=(25, 50),
                     rz_shape=(224,320),
                     is_gray=True):
            super(FgCutmix, self).__init__()
            prob_map = torch.tensor(torch.load(prob_map_pth))  # probability map path

            h, w = prob_map.shape
            self.prob_map = prob_map.resize_(rz_shape)

            if not p_thr:
                self.p_thr = torch.tensor(torch.mean(self.prob_map))
            else:
                self.p_thr = p_thr  # if >p_thr, then generate the patch

            nh, nw = rz_shape
            self.roi_width_range = (int(roi_width_range[0] *nw// w), int(roi_width_range[1]*nw // w))
            self.roi_height_range = (int(roi_height_range[0]*nh // h), int(roi_height_range[1] *nh// h))
            if not area_thr:
                self.area_thr = self.roi_height_range[0] * self.roi_width_range[0] * torch.mean(self.prob_map)
            else:
                self.area_thr = area_thr  # limit the patch area

            self.is_gray = is_gray
            self.imgs_mem = self._load_mem_imgs(imgs_mem_pth)

        def _load_mem_imgs(self, pth):
            ims = torch.load(pth)
            return ims

        def _sel_region_accord_prob(self, prob_map,
                                    p_thr, area_thr,
                                    width_range=(10, 15), height_range=(25, 50)):
            '''generate x,y w,h acording to probability map'''
            h, w = prob_map.shape
            flg = True
            while flg:
                # generate a location
                x = torch.randint(0, w-max(width_range), (1,))[0]
                y = torch.randint(0, h-max(height_range), (1,))[0]
                # x = random.randint(0, w-max(width_range)-5)
                # y = random.randint(0, h-max(height_range)-5)

                # judge the probability in the location
                p = prob_map[y, x]
                if p > p_thr:
                    # generate a random patch width and height
                    w_n = torch.randint(*width_range, (1,))[0]
                    h_n = torch.randint(*height_range, (1,))[0]
                    y_end = y+h_n
                    x_end = x+w_n
                    p_area_sum = torch.sum(prob_map[y:y_end, x:x_end])
                    if p_area_sum > area_thr:
                        flg = False

            return x, y, w_n, h_n

        def _get_rand_box(self, x=0, y=96, w=360, h=78,
                          roi_width_range=(10, 25), roi_height_range=(25, 50)):
            '''get random box in the foreground mask
            Parameters
            ----------
            x : int
                top-left x - coordination for the foreground region
            y : int
                top-left y - coordination for the foreground region
            w : int
                rectangle width for the foreground region
            h : int
                rectangle height for the foreground region
            roi_width_range : tuple, optional
                box width range, by default (10, 25)
            roi_height_range : tuple, optional
                box height range, by default (25, 50)

            Returns
            -------
            (int,int,int,int)
                x,y,w,h
            '''
            random_roi_x = np.random.randint(x, x + w)
            random_roi_y = np.random.randint(y, y + h)
            random_roi_width = np.random.randint(*roi_width_range)
            random_roi_height = np.random.randint(*roi_height_range)

            if random_roi_x + random_roi_width > x + w:
                random_roi_width = x + w - random_roi_x
            if random_roi_y + random_roi_height > y + h:
                random_roi_height = y + h - random_roi_y
            return random_roi_x, random_roi_y, random_roi_width, random_roi_height

        def cutMix_direct(self, x):
            '''
            mix directly
            input:
            x   tensor, b, c, t, h,w
            return:
            x   tensor, pesudo samples, b, c, t, h, w
            y   scalar, pesudo label, range from 0 to 1
            '''
            # get the random region
            # roi_x, roi_y, roi_w, roi_h = self._get_rand_box(*self.mask_cord,
            #                                             roi_width_range=(15, 30),
            #                                             roi_height_range=(30, 50))
            emb_im, roi_x, roi_y, roi_w, roi_h = self._rand_proc_embedImg()

            # print(x.shape, emb_im.shape)

            x = rearrange(x, 'b c t h w -> b t c h w')
            x[:, :, :, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = emb_im

            x = rearrange(x, 'b t c h w -> b c t h w')

            y = self._label_smooth(roi_h, roi_w, x.shape[-2], x.shape[-1])
            return x, y

        def _get_gaussian_mask(self, x_mu, y_mu, x_sigma, y_sigma, h, w):
            '''generate one gaussian mask

            Parameters
            ----------
            x_mu : int
                x mean (location for x )
            y_mu : int
                y mean (location for y)
            x_sigma : float
                variance for gaussian along x-direction
            y_sigma : float
                variance for gaussian along y-direction
            h : int
                region height
            w : int
                region width

            Returns
            -------
            tensor
                2d gaussian mask, (h, w)
            '''
            x, y = torch.arange(h), torch.arange(w)

            # x_mu, y_mu = random.randint(0, w), random.randint(0, h)
            # x_sigma = max(10, int(np.random.uniform(high=max_size) * w))
            # y_sigma = max(10, int(np.random.uniform(high=max_size) * h))

            gx = torch.exp(-(x - x_mu) ** 2 / (2 * x_sigma ** 2))
            gy = torch.exp(-(y - y_mu) ** 2 / (2 * y_sigma ** 2))
            g = torch.outer(gx, gy)
            # g /= np.sum(g)  # normalize, if you want that

            # sum_g = np.sum(g)
            # lam = sum_g / (w * h)
            # print(lam)

            # plt.plot(gx, x, interpolation="nearest", origin="lower")
            # plt.show()
            # g = np.dstack([g, g, g])

            return g

        def smoothCutMix(self, x, alpha=0.5):
            '''
            Ignore the boundaries of the windows and mix the contents with linear fusion strategy.
            mix one image at the same location in the sequence
            return:
            x tensor
            y label, 0 is normal, 1 is anormal, range from 0 to 1
            '''
            # Using linear fusion to fuse corresponding regions

            emb_im, roi_x, roi_y, roi_w, roi_h = self._rand_proc_embedImg()

            # print(x.shape, emb_im.shape)

            x = rearrange(x, 'b c t h w -> b t c h w')

            cutPatch = x[:, :, :, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            mixed = alpha * cutPatch + (1 - alpha) * emb_im.squeeze(0)
            x[:, :, :, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mixed

            x = rearrange(x, 'b t c h w -> b c t h w')

            y = self._label_smooth(roi_h, roi_w, x.shape[-2], x.shape[-1])

            return x, y

        def _proc_embedIm(self, emb_im):
            '''processing the embed image according to a random selected region

            Parameters
            ----------
            emb_im : tensor, c, h, w
                embeded image

            Returns
            -------
            emb_im,  resized embeded image
            x, y, roi_w, roi_h: left-top-x, left-top y coordination and the width and height of the rectangle
            '''
            roi_x, roi_y, roi_w, roi_h = self._sel_region_accord_prob(self.prob_map, self.p_thr, self.area_thr,
                                                                      width_range=self.roi_width_range,
                                                                      height_range=self.roi_height_range
                                                                      )
            # random select one image
            # ind = torch.randint(0, len(self.imgs_mem), (1,))[0]
            # emb_im = self.imgs_mem[ind]

            # resize the image to the same size
            rz = transforms.Resize((roi_h, roi_w))
            emb_im = rz(emb_im)

            # if ped, embeded into a gray image
            if self.is_gray:
                emb_im = torch.mean(emb_im, dim=0, keepdim=True)

            return emb_im, roi_x, roi_y, roi_w, roi_h

        def _rand_proc_embedImg(self):
            '''select a region according to probability map, and resize the embed image
            based on the selected region.

            Returns
            -------
            emb_im,  resized embeded image
            x, y, roi_w, roi_h: left-top-x, left-top y coordination and the width and height of the rectangle
            '''
            roi_x, roi_y, roi_w, roi_h = self._sel_region_accord_prob(self.prob_map, self.p_thr, self.area_thr,
                                                                      width_range=self.roi_width_range,
                                                                      height_range=self.roi_height_range
                                                                      )
            # random select one image
            ind = torch.randint(0, len(self.imgs_mem), (1,))[0]
            emb_im = self.imgs_mem[ind]

            # resize the image to the same size
            rz = transforms.Resize((roi_h, roi_w))
            emb_im = rz(emb_im)

            # if ped, embeded into a gray image
            if self.is_gray:
                emb_im = torch.mean(emb_im, dim=0, keepdim=True)

            return emb_im, roi_x, roi_y, roi_w, roi_h

        def _label_smooth(self, roi_h, roi_w, im_h, im_w):
            # 0 is normal data, and 1 is anomaly
            lam = roi_h * roi_w / (im_h * im_w)
            # y = λyi + (1 − λ)yj
            y = 1 - lam
            return y

        def _gen_roi_mask(self, roi_h, roi_w, msk_type='gaussian'):
            '''generate the mask acoording to the window size

            Parameters
            ----------
            roi_h : int
                roi window height
            roi_w : int
                roi window width
            msk_type : str, optional
                mask type, by default 'gaussian'

            Returns
            -------
            tensor
                mask
            '''
            x_mu, y_mu = random.randint(0, roi_h), random.randint(0, roi_w)
            x_sigma = max(10, int(torch.rand(1)[0] * roi_h))
            y_sigma = max(10, int(torch.rand(1)[0] * roi_w))
            if msk_type == 'gaussian':
                mask = self._get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, roi_h, roi_w)
            return mask

        def _preproc_same_embIm_mix(self, emb_im, msk_type='gaussian'):
            '''preprocessing for mix

            Parameters
            ----------
            emb_im : tensor h,w,c
                _description_

            Returns
            -------
            _type_
                _description_
            '''
            emb_im, roi_x, roi_y, roi_w, roi_h = self._proc_embedIm(emb_im=emb_im)
            mask = self._gen_roi_mask(roi_h, roi_w, msk_type=msk_type)
            return emb_im, mask, roi_x, roi_y, roi_h, roi_w

        def gaussianSmoothCutMix(self, x, max_size=0.5, emb_type='same'):
            '''
            creat mask using gaussian, and use it to mix the sequence and embeded image
            input:
            x          b, c, t, h, w
            max_size   the variance radio for gaussian
            emb_type   str, same or diffLoc or diffLocImg

            return:
            x   tensor, pesudo samples, b, c, t, h, w
            y   scalar, pesudo label, range from 0 to 1
            '''
            # get the random region
            assert 1 > max_size > 0, 'max_size is between 0 and 1'
            # random select one image
            ind = torch.randint(0, len(self.imgs_mem), (1,))[0]
            emb_im = self.imgs_mem[ind]

            # emb_im, roi_x, roi_y, roi_w, roi_h = self._proc_embedIm(emb_im=emb_im)

            # # print(x.shape, emb_im.shape)

            # # ------------------mix----------------
            # # 1. create the mask
            # mask = self._gen_roi_mask(roi_h,roi_w, msk_type='gaussian')

            # 2. get random region from sequences
            x = rearrange(x, 'b c t h w -> b t c h w')

            roi_area = 0
            b, t, c, h, w = x.shape

            # embeded the same image in the sequence
            if emb_type == 'same':
                # prepare
                emb_im, mask, roi_x, roi_y, roi_h, roi_w = self._preproc_same_embIm_mix(emb_im=emb_im,
                                                                                        msk_type='gaussian')
                emb_im = emb_im.type_as(x)
                mask = mask.type_as(x)
                # 2. get the region
                cutPatch = x[:, :, :, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                # 3. mix the two
                # print(f'mask shape:{mask.shape}, cutpatch shape:{cutPatch.shape}' )
                mixed = mask * emb_im + (1 - mask) * cutPatch

                # mixed = alpha*cutPatch+(1-alpha)*emb_im.squeeze(0)
                x[:, :, :, roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mixed

                roi_area = roi_area + roi_h * roi_w * b * t
            # randomly embed a same image at different location in the sequence
            if emb_type == 'diffLoc':
                for i in range(b):
                    for j in range(t):
                        # emb_im, roi_x, roi_y, roi_w, roi_h = self._proc_embedIm(emb_im)

                        # # ------------------mix----------------
                        # # 1. create the mask
                        # mask = self._get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, roi_h, roi_w)
                        emb_im, mask, roi_x, roi_y, roi_h, roi_w = self._preproc_same_embIm_mix(emb_im=emb_im,
                                                                                                msk_type='gaussian')
                        emb_im = emb_im.type_as(x)
                        mask = mask.type_as(x)
                        # 2. get the region
                        cutPatch = x[i][j][..., roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                        # 3. mix the two
                        # print(f'mask:{mask.shape}, cutpatch:{cutPatch.shape}, embim:{emb_im.shape}')
                        mixed = mask * emb_im + (1 - mask) * cutPatch
                        x[i][j][..., roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mixed
                        roi_area = roi_area + roi_h * roi_w

            # randomly embed a different image at different location in the sequence
            if emb_type == 'diffLocImg':
                for i in range(b):
                    for j in range(t):
                        emb_im, roi_x, roi_y, roi_w, roi_h = self._rand_proc_embedImg()

                        # ------------------mix----------------
                        # 1. create the mask
                        mask = self._gen_roi_mask(roi_h, roi_w)
                        emb_im = emb_im.type_as(x)
                        mask = mask.type_as(x)
                        # 2. get the region
                        cutPatch = x[i][j][..., roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                        # 3. mix the two
                        # print(f'mask:{mask.shape}, cutpatch:{cutPatch.shape}, embim:{emb_im.shape}')
                        mixed = mask * emb_im + (1 - mask) * cutPatch
                        x[i][j][..., roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mixed

                        roi_area = roi_area + roi_w * roi_h
            x = rearrange(x, 'b t c h w -> b c t h w')
            lamb = roi_area / (b * t * h * w)
            y = torch.ones(x.shape[0],) - lamb
            return x, y


