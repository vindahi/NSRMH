from scipy.linalg import hadamard
from network import *
import os
import torch
import argparse
from losses import *
from utilss import *
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--hash_dim', type=int, default=32)
parser.add_argument('--noise_rate', type=float, default=0.5)
parser.add_argument('--dataset', type=str, default='flickr')
parser.add_argument('--num_gradual', type=int, default=100)
parser.add_argument('--Lambda', type=float, default=0.6)
parser.add_argument('--optimizer_lr', type=float, default=1e-5)
parser.add_argument('--optimizer_weight_decay', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--random_state', type=int, default=1)
parser.add_argument('--threshold_rate', type=float, default=0.3)
parser.add_argument('--image_dim', type=int, default=512)
parser.add_argument('--text_dim', type=int, default=512)
parser.add_argument('--classes', type=int, default=24)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128])
parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128])
parser.add_argument('--mlpdrop', type=float, default=0.01)
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--param_cluster', type=float, default=0.01)
parser.add_argument('--param_it', type=float, default=1)
parser.add_argument('--param_sup', type=float, default=0.0001)
parser.add_argument('--param_sign', type=float, default=0.01)
parser.add_argument('--param_sim', type=float, default=1)

args = parser.parse_args()

args.img_hidden_dim.insert(0, args.image_dim)
args.txt_hidden_dim.insert(0, args.text_dim)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda=args.Lambda
num_gradual = args.num_gradual
torch.multiprocessing.set_sharing_strategy('file_system')
loss_l2 = nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

logName = args.dataset + '_' + str(args.hash_dim) + '_' + str(args.noise_rate)
log = logger(logName)

def train(args, bit):
    device = torch.device("cuda:0")
    log.info('Loading data')
    log.info('param_sim:  %.6f', args.param_sim)
    log.info('param_it:  %.6f', args.param_it)
    log.info('param_sign:  %.6f', args.param_sign)
    log.info('param_sup:  %.6f', args.param_sup)


    train_loader, test_loader, dataset_loader = get_data(args)
    
    fusionnet = FusionNwt(args).to(device)

    # 初始化权重 W
    W = torch.empty(args.classes, bit).to(device)  # 直接在CUDA上创建
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.nn.Parameter(W, requires_grad=True)  # 设置为可训练参数
    fusionnet.register_parameter('W', W)  # 注册 W 到图像网络
    
    fusionnet.train()

    optimizer = torch.optim.Adam(fusionnet.parameters(), lr=args.lr, weight_decay=1e-5)

    bestmap = 0.0
    for epoch in range(args.epoch):
        train_loss = 0.0
        
        if (epoch + 1) % 2 == 0:

            fusionnet.eval()

            re_B, re_L = computemultimodal_result(dataset_loader, fusionnet, device=device)
            qu_B, qu_L = computemultimodal_result(test_loader, fusionnet, device=device)
            MAP50 = calculate_top_map(qu_B=qu_B.numpy(), re_B=re_B.numpy(), qu_L=qu_L.numpy(), re_L=re_L.numpy(), topk=50)
            log.info('MAP50: %.4f', MAP50)
            if MAP50 > bestmap:
                bestmap = MAP50
                _dict = {
                    'retrieval_B': re_B.cpu().numpy().astype(np.int8),
                    'val_B': qu_B.cpu().numpy().astype(np.int8),
                    'L_db': re_L.cpu().numpy(),
                    'L_te': qu_L.cpu().numpy()
                }
                sava_path = 'Hashcode/Ours_' + str(args.hash_dim) + '_' + args.dataset + '_bits.mat'
                sio.savemat(sava_path, _dict)
                data = np.vstack((re_B.cpu().numpy(), qu_B.cpu().numpy()))
                label_data = np.vstack((re_L.cpu().numpy(), qu_L.cpu().numpy()))
                label_indices = np.argmax(label_data, axis=1)

                # Perform t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                Y = tsne.fit_transform(data)

                # Create the plot
                plt.figure(figsize=(8, 8))
                scatter = plt.scatter(Y[:, 0], Y[:, 1], c=label_indices, cmap='jet', alpha=0.5)
                plt.xlim([-70, 70])
                plt.ylim([-70, 70])
                plt.title(f't-SNE Visualization: Round {epoch}, HashDim {args.hash_dim}')
                plt.colorbar(scatter, label='Class Labels')

                # Save the plot
                plt_path = f'TSNE/tsne1_round_{epoch}_noise_{args.noise_rate}_hashdim_{args.hash_dim}.png'
                plt.savefig(plt_path)
                plt.close()
                log.info('best_map: %.4f' % (bestmap))
                print("best_map: %.4f" % (bestmap))
            else:
                _dict = {
                    'retrieval_B': re_B.cpu().numpy().astype(np.int8),
                    'val_B': qu_B.cpu().numpy().astype(np.int8),
                    'L_db': re_L.cpu().numpy(),
                    'L_te': qu_L.cpu().numpy()
                }
                sava_path = 'Hashcode/Ours_' + str(args.hash_dim) + '_' + args.dataset + '_bits.mat'
                sio.savemat(sava_path, _dict)
                data = np.vstack((re_B.cpu().numpy(), qu_B.cpu().numpy()))
                label_data = np.vstack((re_L.cpu().numpy(), qu_L.cpu().numpy()))
                label_indices = np.argmax(label_data, axis=1)

                # Perform t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                Y = tsne.fit_transform(data)

                # Create the plot
                plt.figure(figsize=(8, 8))
                scatter = plt.scatter(Y[:, 0], Y[:, 1], c=label_indices, cmap='jet', alpha=0.5)
                plt.xlim([-70, 70])
                plt.ylim([-70, 70])
                plt.title(f't-SNE Visualization: Round {epoch}, HashDim {args.hash_dim}')
                plt.colorbar(scatter, label='Class Labels')

                # Save the plot
                plt_path = f'TSNE/tsne2_round_{epoch}_noise_{args.noise_rate}_hashdim_{args.hash_dim}.png'
                plt.savefig(plt_path)
                plt.close()
                log.info('best_map: %.4f' % (bestmap))
                print("best_map: %.4f" % (bestmap))

        pred = Improvedmulti_get_loss(fusionnet, train_loader, args.threshold_rate, epoch, W)
        for image, tag, tlabel, label, ind in train_loader:
            current_pred = pred[ind]
            clean_samples = current_pred == 1
            
            if clean_samples.sum() > 0:
                image = image[clean_samples].to(device).float()
                tag = tag[clean_samples].to(device).float()
                label = label[clean_samples].to(device)

                _, aff_norm, aff_label = affinity_tag_multi(label.cpu().numpy(), label.cpu().numpy())
                aff_label = torch.Tensor(aff_label).to(device)  # 将相似度标签转回CUDA
                
                optimizer.zero_grad()

                code, predcode = fusionnet(image, tag)

                recon_loss = loss_l2(torch.sigmoid(predcode), label.float())

                H_norm = F.normalize(code)
                similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)

                center = fusionnet.centroids.to(dtype=torch.float32).cuda()


                code_center = code.mm(center.t())
                label = label.to(torch.float)  # 将 label 转换为 Float 类型
                constr_loss = bce_loss(code_center, label)


                similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)

                code_cen_loss = code_center_loss(code, center, label)

                losssign = loss_l2(code, torch.sign(code))

                loss = similarity_loss  * args.param_sim + recon_loss + code_cen_loss * args.param_it + constr_loss * args.param_sup + losssign * args.param_sign
                train_loss += loss.item()  # 使用 item() 获取标量值
                loss.backward()
                optimizer.step()
        
        train_loss /= len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        log.info("epoch:%d, bit:%d, dataset:%s, noise_rate:%.1f, loss:%.3f, best_map:%.4f" % (
                epoch + 1, bit, args.dataset, args.noise_rate, train_loss, bestmap))
    
    return fusionnet



def eval(moodel, args):
    moodel.eval()
    device = torch.device("cuda:0")
    train_loader, test_loader, dataset_loader = get_data(args)
    re_B, re_L = computemultimodal_result(dataset_loader, moodel, device=device)
    qu_B, qu_L = computemultimodal_result(test_loader, moodel, device=device)
    MAP50 = calculate_top_map(qu_B=qu_B.numpy(), re_B=re_B.numpy(), qu_L=qu_L.numpy(), re_L=re_L.numpy(), topk=50)
    print('MAP50:', MAP50)

    return 0

    

if __name__ == "__main__":
    seed_setting(args.seed)
    trianmodel = train(args, args.hash_dim)
    eval(trianmodel, args)




