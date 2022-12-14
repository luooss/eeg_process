import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    # select self loop edges
    inv_mask = mask.logical_not()
    loop_weight = torch.full((num_nodes, ), fill_value, dtype=None if edge_weight is None else edge_weight.dtype, device=edge_weight.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1).to(edge_index.device)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm2(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        # num_nodes = x.size(0)
        # if edge_weight is None:
        #     edge_weight = torch.ones((edge_index.size(1),),
        #                              dtype=dtype,
        #                              device=edge_index.device)

        # fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        # scatter_add_ on zeros
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """
            x: torch.Size([num_graphs*num_nodes, num_node_features])
            edge_index: torch.Size([2, sum of num_edges for each graph])
            edge_weight: torch.Size([num_graphs*num_nodes*num_nodes])
        """
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.cached_result, self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        # num_edges = num_nodes * num_nodes
        # num_edgeweights_tolearn = (num_nodes * num_nodes + num_nodes) / 2
        # torch.Size([num_edgeweights_tolearn])
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = nn.Linear(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2)

    def forward(self, data, alpha=0):
        # data: torch.geometric.data.Batch
        # num_graphs
        batch_size = len(data.y)
        # x: torch.Size([num_graphs*num_nodes, num_node_features])
        # edge_index: torch.Size([2, sum of num_edges for each graph])
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        # torch.Size([num_graphs*num_nodes*num_nodes])
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        
        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x, domain_output


class GradientReversalFuntion(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grads):
        dx = ctx.lambda_ * grads.neg()
        return dx, None


# with changing lambda
class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_):
        return GradientReversalFuntion.apply(x, lambda_)


class SGCFeatureExtractor(nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hidden, K):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hidden: hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SGCFeatureExtractor, self).__init__()
        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.num_features = num_features
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        # num_edges = num_nodes * num_nodes
        # num_edgeweights_tolearn = (num_nodes * num_nodes + num_nodes) / 2
        # torch.Size([num_edgeweights_tolearn])
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hidden, K=K)

    def forward(self, data):
        # data: torch.geometric.data.Batch
        # num_graphs
        batch_size = len(data.y)
        # x: torch.Size([num_graphs*num_nodes, num_node_features])
        # edge_index: torch.Size([2, sum of num_edges for each graph])
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        # torch.Size([num_graphs*num_nodes*num_nodes])
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = self.bn1(x)
        x_smooth, xz = self.conv1(x, edge_index, edge_weight)
        xz = F.leaky_relu(self.bn2(xz))
        # self-attention, concate pooling
        xz = xz.view(batch_size, self.num_nodes, self.num_hidden)
        xz_t = torch.transpose(xz, 1, 2)
        attention = torch.softmax(torch.bmm(xz, xz_t), dim=2)
        x_attention = torch.bmm(attention, x_smooth.view(batch_size, self.num_nodes, self.num_features))
        # (batch_size, num_nodes*num_features)
        x_attention = x_attention.view(batch_size, -1)
        return x_attention


class LabelClassifier(nn.Module):
    def __init__(self, input_dim, dropout):
        super(LabelClassifier, self).__init__()
        # if you want to apply additional operations in between layers, wirte them separately
        # or use nn.Sequential()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim//2, 3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, dropout):
        super(DomainClassifier, self).__init__()

        self.gradrev = GradientReversalLayer()

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.bn = nn.BatchNorm1d(input_dim//2)
        self.dropout = dropout
        self.fc2 = nn.Linear(input_dim//2, 2)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x, lambda_):
        x = self.gradrev(x, lambda_)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.lsm(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(AutoEncoder, self).__init__()

        # Encoder
        # eye (23)
        # 1s (62, 5, 1) -> (62*3)
        # 3s (62, 5, 3) -> (62*13)
        # 5s (62, 5, 5) -> (62*20)
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim//2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim//4),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim//4, input_dim//2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim),
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded