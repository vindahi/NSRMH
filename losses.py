import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F



def advanced_split_prob(prob, threshold):
        # 使用多模态信息和损失分布
        median_prob = np.median(prob)
        threshold = max(threshold, median_prob)
        
        # 动态阈值
        pred = (prob > threshold).astype(int)
        
        # 置信度加权
        confidence_weight = np.abs(prob - 0.5) * 2
        pred = (pred * confidence_weight > 0.5).astype(int)
        
        return pred



def Improvedmulti_get_loss(fusnet, data_loader, Threshold, epoch, W):
    sample_losses = []
    
    for image, tag, tlabel, label, ind in data_loader:
        image = image.to('cuda').float()
        tag = tag.to('cuda').float()
        label = label.to('cuda')
        tlabel = tlabel.to('cuda')
        
        code, _ = fusnet(image, tag)
        
        with torch.no_grad():
            y = label.float()
            inverted_y = torch.add(torch.mul(y, -1.), 1.)
            
            features_sims = code @ W.tanh().t()
        
            select_examples = torch.mul(y.unsqueeze(1), inverted_y.unsqueeze(0))
            
            features_errors = torch.abs(features_sims.unsqueeze(1) - features_sims.unsqueeze(0))
            
            paired_errors = features_errors * select_examples
            
            sample_error = paired_errors.mean(dim=1)
            
            # 正确标签判断
            right = ((tlabel==label).float().mean(1) == 1).float()
            
            for i in range(len(sample_error)):
                sample_losses.append((
                    ind[i].item(), 
                    float(sample_error[i].mean()),  # 使用float()转换 
                    float(right[i])
                ))
    
    sample_losses_sorted = sorted(sample_losses, key=lambda x: x[0])
    sorted_losses = [item[1] for item in sample_losses_sorted]
    sorted_losses = np.array(sorted_losses)
    
    sorted_losses = (sorted_losses - sorted_losses.min() + 1e-8) / \
                    (sorted_losses.max() - sorted_losses.min() + 1e-8)
    sorted_losses = sorted_losses.reshape(-1, 1)
    
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=5e-1, reg_covar=5e-4)
    gmm.fit(sorted_losses)
    prob = gmm.predict_proba(sorted_losses)
    prob = prob[:, gmm.means_.argmin()]
    
    if epoch+1 >= 20:
        pred = advanced_split_prob(prob, Threshold)
    else:
        pred = advanced_split_prob(prob, 0)
      
    return torch.Tensor(pred)