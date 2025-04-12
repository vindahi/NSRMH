import torch.nn as nn
from torch.nn import functional as F
import torch

class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512], act=nn.Tanh(), dropout=0.01):
        super(MLP, self).__init__()

        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim[-1]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  

        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        self.activations.append(act)
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))

        for i in range(len(self.hidden_dim) - 1):
            self.layers.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.activations.append(act)
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i + 1])) 

        self.layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer, activation, batch_norm in zip(self.layers, self.activations, self.batch_norms):
            x = layer(x)
            x = activation(x)
            x = batch_norm(x)
            x = self.dropout(x)

        return x



class FusionNwt(nn.Module):
    def __init__(self, args):
        super(FusionNwt, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.hash_dim)
        self.classes = args.classes
        self.batch_size = args.batch_size
        
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)
        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)


        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.classify = nn.Linear(self.nbit, self.classes)
        self.centroids = nn.Parameter(torch.randn(self.classes, self.nbit)).to(dtype=torch.float32)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)#nbit length
        textH = self.textMLP(text)
        nec_vec = imageH + textH   
        code = self.hash_output(nec_vec)   
        return code, self.classify(code)


