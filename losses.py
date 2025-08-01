import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from typing import List, Tuple, Any

def advanced_split_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    """
    使用多模态信息和损失分布进行样本划分
    参数:
        prob: 样本属于干净分布的概率 [N]
        threshold: 基础划分阈值
    返回:
        预测标签 (1=干净, 0=噪声)
    """
    # 动态阈值设置：确保不低于中位数
    median_prob = np.median(prob)
    dynamic_threshold = max(threshold, median_prob)
  
    # 基础预测
    base_pred = (prob > dynamic_threshold).astype(int)
  
    # 置信度加权：离决策边界越远置信度越高
    confidence_weight = np.abs(prob - 0.5) * 2  # [0,1] -> [0,1]
    weighted_pred = base_pred * confidence_weight
  
    # 加权决策
    final_pred = (weighted_pred > 0.5).astype(int)
  
    # 边界保护：防止全0/全1输出
    if np.all(final_pred == 0):
        final_pred[np.argmax(prob)] = 1
    elif np.all(final_pred == 1):
        final_pred[np.argmin(prob)] = 0
      
    return final_pred

def compute_pairwise_errors(features_sims: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算样本间特征相似度误差矩阵
    参数:
        features_sims: 特征相似度矩阵 [B, D]
        y: 样本标签 [B]
    返回:
        样本误差向量 [B]
    """
    inverted_y = 1.0 - y  # 标签取反
    select_mask = y.unsqueeze(1) * inverted_y.unsqueeze(0)  # 正负样本对掩码
  
    # 计算特征差异
    feature_diffs = torch.abs(
        features_sims.unsqueeze(1) - features_sims.unsqueeze(0)
    )  # [B, B, D]
  
    # 掩码处理并聚合误差
    masked_errors = feature_diffs * select_mask.unsqueeze(-1)
    sample_errors = masked_errors.sum(dim=(1, 2)) / (select_mask.sum(dim=1) + 1e-8)
  
    return sample_errors

def normalize_losses(losses: List[float]) -> np.ndarray:
    """
    归一化损失值到[0,1]区间
    参数:
        losses: 原始损失值列表
    返回:
        归一化后的损失数组 [N, 1]
    """
    losses_arr = np.array(losses)
    min_val, max_val = losses_arr.min(), losses_arr.max()
  
    # 处理全等值情况
    if max_val - min_val < 1e-6:
        return np.zeros_like(losses_arr).reshape(-1, 1)
  
    return ((losses_arr - min_val) / (max_val - min_val + 1e-8)).reshape(-1, 1)

def Improvedmulti_get_loss(
    fusnet: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    threshold: float,
    epoch: int,
    W: torch.Tensor
) -> torch.Tensor:
    """
    改进的多模态损失计算与噪声样本检测
    参数:
        fusnet: 多模态融合网络
        data_loader: 数据加载器
        threshold: 噪声检测阈值
        epoch: 当前训练轮次
        W: 可学习的权重矩阵
    返回:
        样本预测标签 (1=干净, 0=噪声)
    """
    sample_losses = []  # (index, loss, is_correct)
    device = next(fusnet.parameters()).device
  
    # 确保W在正确设备上
    W = W.detach().to(device).requires_grad_(False)
  
    fusnet.eval()
    with torch.no_grad():
        for batch in data_loader:
            # 解包数据 (适配不同结构的数据加载器)
            if len(batch) == 5:
                image, tag, tlabel, label, ind = batch
            else:
                raise ValueError("数据加载器应返回5元素元组")
              
            image = image.to(device).float()
            tag = tag.to(device).float()
            label = label.to(device)
            tlabel = tlabel.to(device)
          
            # 前向传播
            code, _ = fusnet(image, tag)
          
            # 计算特征相似度
            features_sims = torch.tanh(code @ W.t())  # [B, D]
          
            # 计算样本误差
            y = label.float()
            sample_errors = compute_pairwise_errors(features_sims, y)
          
            # 正确标签判断
            is_correct = ((tlabel == label).float().mean(dim=1) == 1).float()
          
            # 存储结果
            for i in range(len(sample_errors)):
                sample_losses.append((
                    ind[i].item(),
                    sample_errors[i].item(),
                    is_correct[i].item()
                ))
  
    # 按索引排序
    sample_losses.sort(key=lambda x: x[0])
    sorted_losses = [loss for _, loss, _ in sample_losses]
  
    # 归一化损失
    norm_losses = normalize_losses(sorted_losses)
  
    # 二元GMM拟合
    gmm = GaussianMixture(
        n_components=2,
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-4
    )
    gmm.fit(norm_losses)
  
    # 获取干净样本的概率
    prob = gmm.predict_proba(norm_losses)
    clean_component = gmm.means_.argmin()
    clean_probs = prob[:, clean_component]
  
    # 动态阈值策略
    use_threshold = threshold if epoch >= 19 else 0.0  # 前20轮使用宽松阈值
    pred = advanced_split_prob(clean_probs, use_threshold)
  
    return torch.tensor(pred, dtype=torch.float32)
