import numpy as np
import h5py
import torch.utils.data as util_data
from torchvision import transforms
import torch
from tqdm import tqdm
import scipy.io as sio
import logging
import os
import random
import os.path as osp

class DataList(object):
    def __init__(self, dataset, data_type, transform, noise_type, noise_rate, random_state):
        self.data_type = data_type
        if dataset == 'nuswide':
            data = sio.loadmat('./data/nus_clip_all.mat')
            noise = h5py.File('./noise/nus-lall-noise_{}.h5'.format(noise_rate))
        elif dataset == 'flickr':
            data = sio.loadmat('./data/mir_clip_all.mat')
            noise = h5py.File('noise/mir-lalll-noise_{}.h5'.format(noise_rate))
        elif dataset == 'coco':
            data = sio.loadmat('./data/coco_clip_all.mat')
            noise = h5py.File('./noise/coco-lall-noise_{}.h5'.format(noise_rate))
        elif dataset == 'wiki':
            data = sio.loadmat('./data/wiki_data_all.mat')
            noise = h5py.File('./noise/wiki-lall-noise_{}.h5'.format(noise_rate))
        if data_type == "train":
            fi = list(data['I_tr'])
            fl = list(data['L_tr'])
            ffl = list(noise['result'])
            ft = list(data['T_tr'])
            self.imgs = fi
            self.labs = fl
            self.flabs = ffl
            self.tags = ft
            lab = self.labs[1]
            lab = lab.astype(int)
        elif data_type == "test":
            fi = list(data['I_te'])
            fl = list(data['L_te'])
            ft = list(data['T_te'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        elif data_type == "database":
            fi = list(data['I_db'])
            fl = list(data['L_db'])
            ft = list(data['T_db'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img.astype(np.float32)
        lab = self.labs[index]
        lab = lab.astype(int)
        tlab = lab
        if self.data_type == "train":
            lab = self.flabs[index]
            lab = lab.astype(int)
        tag = self.tags[index]
        tag = tag.astype(np.float32)
        return img, tag, tlab, lab, index

    def __len__(self):
        return len(self.imgs)







def get_data(args):
    dsets = {}
    dset_loaders = {}

    for data_type in ["train", "test", "database"]:
        dsets[data_type] = DataList(args.dataset, data_type,
                                   transforms.ToTensor(), args.noise_type, args.noise_rate, args.random_state)
        print(data_type, len(dsets[data_type]))
        dset_loaders[data_type] = util_data.DataLoader(dsets[data_type],
                                                       batch_size=args.batch_size,
                                                       shuffle=True, num_workers=2)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"]


def compute_model_result(dataloader, image_net, text_net, device):
    bsimage, bstext, clses = [], [], []
    image_net.eval()
    text_net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bsimage.append(image_net(img.to('cuda')).data.cpu())
        bstext.append(text_net(tag.to('cuda')).data.cpu())

    return torch.cat(bsimage).sign(), torch.cat(bstext).sign(), torch.cat(clses)

def compute_modelpred_result(dataloader, image_net, text_net, device):
    bsimage, bstext, clses = [], [], []
    image_net.eval()
    text_net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs,_ = image_net(img.to('cuda'))
        bt,_ = text_net(tag.to('cuda'))
        bstext.append(bt.data.cpu())
        bsimage.append(bs.data.cpu())


    return torch.cat(bsimage).sign(), torch.cat(bstext).sign(), torch.cat(clses)


def computemultimodal_result(dataloader, multimodel, device):
    bsmulti, clses = [], []
    multimodel.eval()

    for img, tag, tcls, cls, _ in tqdm(dataloader):
        clses.append(cls)
        code, _= multimodel(img.to('cuda'), tag.to('cuda'))
        bsmulti.append(code.data.cpu())
    
    return torch.cat(bsmulti).sign(), torch.cat(clses)


def compute_modelsignal_result(dataloader, crossmodel, device):
    bsimage, bstext, clses = [], [], []
    # image_net.eval()
    # text_net.eval()
    crossmodel.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        clses.append(cls)
        # bsimage.append(image_net(img.to('cuda')).data.cpu())
        # bstext.append(text_net(tag.to('cuda')).data.cpu())
        bs, bt, _, _ = crossmodel(img.to('cuda'), tag.to('cuda'))
        bstext.append(bt.data.cpu())
        bsimage.append(bs.data.cpu())


    return torch.cat(bsimage).sign(), torch.cat(bstext).sign(), torch.cat(clses)


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, int(tsum)) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    log_name = str(fileName) + '.log'
    log_dir = './logalb'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def zero2eps(x):
    x[x == 0] = 1
    return x

# Refer
def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

# Check in 2022-1-3
def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    '''
    Use label or plabel to create the graph.
    :param tag1:
    :param tag2:
    :return:
    '''
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    # affinity_matrix[affinity_matrix > 1] = 1
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)

    return in_aff, out_aff, affinity_matrix


def code_center_loss(hash_code, center, label, eps = 1e-5):
    code_length = hash_code.shape[1]
    logit_ii = hash_code.mm(center.t()) / code_length
    our_logit_ii = torch.exp(logit_ii) * label
    mu_logit_ii = (torch.exp(logit_ii) * (1 - label)).sum(1).view(-1, 1).expand(logit_ii.shape[0], logit_ii.shape[1]) + our_logit_ii
    lossi = -((torch.log((our_logit_ii) / (mu_logit_ii + eps) + eps) * label).sum(1) / label.sum(1))
    loss = lossi.mean()
    return loss



def center_loss(centroids):
    centroids_dist = torch.cdist(centroids, centroids, p=2) / centroids.shape[1]
    triu_dist = torch.triu(centroids_dist, diagonal=1)

    # 计算非零的平均距离
    non_zero_dist = triu_dist[triu_dist > 0]
    mean_dist = torch.mean(non_zero_dist) if non_zero_dist.numel() > 0 else 0

    # 处理距离为0的情况
    min_dist = torch.min(triu_dist[triu_dist > 0]) if non_zero_dist.numel() > 0 else 0

    # 将损失设为正值
    reg_term = mean_dist + min_dist
    return reg_term
