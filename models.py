from torch.nn import Parameter
from util import *
from transformer import *
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def write_matrix(matrix_a, files):
    matrix = matrix_a.cpu()
    matrix = matrix.detach().numpy()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            files.write(str(matrix[i][j]) + ",")
        files.write("\n")

class GATNResnet(nn.Module):
    def __init__(self, model_name, num_classes, in_channel=300, t1=0.0, adj_file=None):
        super(GATNResnet, self).__init__()

        # Create multiple backbones
        model = load_model(model_name)
        self.backbone = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # Graph Convolutions
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        
        # Topology
        if num_classes == 20:
            s_adj = gen_A_correlation(num_classes, 1.0, 0.4, "./model/topology/voc_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(num_classes, 0.4, 0.2, "./model/topology/voc_correlation_adj.pkl")
        else:
            s_adj = gen_A_correlation(num_classes, 1.0, 0.2, "./model/topology/coco_correlation_adj.pkl")
            s_adj1 = gen_A_correlation(num_classes, 0.4, 0.2, "./model/topology/coco_correlation_adj.pkl")
            # s_adj = gen_A(num_classes, 0.4, 'data/coco/coco_adj.pkl')
        s_adj = torch.from_numpy(s_adj).type(torch.FloatTensor)
        A_Tensor = s_adj.unsqueeze(-1)
        s_adj1 = torch.from_numpy(s_adj1).type(torch.FloatTensor)
        A_Tensor1 = s_adj1.unsqueeze(-1)


        self.transformerblock = AttentionBlock(num_classes)# TransformerBlock()

        self.linear_A = nn.Linear(80, 80)
        self.A_1 = A_Tensor.permute(2,0,1)
        self.A_2 = A_Tensor1.permute(2,0,1)
        self.A = A_Tensor.unsqueeze(0).permute(0,3,1,2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp, label = None, iters = 1, max_iters = 256600):
        batch_size = feature.shape[0]

        feature = self.backbone(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        
        inp = inp[0]
        # print("self.A_1",self.A_1.shape)
        adj, _ = self.transformerblock(self.A_1.cuda(), self.A_2.cuda())
        # adj = self.A_1.cuda()
        # print("adj_shape",adj.shape)
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = gen_adj(adj)

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        return x

    def get_config_optim(self, lr, lrp, lrt):
        config_optim = []
        config_optim.append({'params': self.backbone.parameters(), 'lr': lr * lrp})
        config_optim.append({'params': self.transformerblock.parameters(), 'lr': lr * lrt})
        config_optim.append({'params': self.gc1.parameters(), 'lr': lr})
        config_optim.append({'params': self.gc2.parameters(), 'lr': lr})
        return config_optim



def gatn_resnet(num_classes, t1, pretrained=True, adj_file=None, in_channel=300):
    return GATNResnet('resnext101_32x16d_swsl', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)
    # return GATNResnet('resnet101', num_classes, t1=t1, adj_file=adj_file, in_channel=in_channel)