import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
from torch_sparse import SparseTensor, spmm

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", batch_norm=True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.use_batch_norm = batch_norm
        if self.use_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
            for layer in range(num_layer):
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if self.use_batch_norm:
                h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_representation

class GNN_Encoder(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim,  JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",batch_norm=True):
        super(GNN_Encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_workers = 2

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type,batch_norm=batch_norm)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")


        self.encoder = GraphTransformerEncoder(embed_dim=emb_dim,gnn_type=gnn_type,k_hop=2)


    def from_pretrained(self, model_file, gpu_id):
        if torch.cuda.is_available():
            self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:' + str(gpu_id)))
        else:
            self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        subgraph_node_index = data.subgraph_node_idx
        subgraph_edge_index = data.subgraph_edge_index
        subgraph_indicator_index = data.subgraph_indicator
        subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") else None
        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None

        node_representation = self.gnn(x, edge_index, edge_attr)

        output = self.encoder(
            x,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=None,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=False
        )

        representation = 0.9*node_representation+0.1*output
        graph_representation = self.pool(node_representation, batch)
        subg_representation = self.pool(output,batch)
        gs_representation = self.pool(representation,batch)

        return graph_representation, subg_representation , gs_representation

class GraphTransformerEncoder(torch.nn.Module):
    def __init__(self,embed_dim,gnn_type="gin",k_hop=2):
        super(GraphTransformerEncoder, self).__init__()
        self.khop_structure_extractor = KHopStructureExtractor(embed_dim, gnn_type=gnn_type, num_layers=k_hop)
        self.norm = torch.nn.BatchNorm1d(embed_dim)
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = self.khop_structure_extractor(
            x=x,
            edge_index=edge_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_attr=subgraph_edge_attr,
        )
        if self.norm is not None:
            output = self.norm(output)
        return output

class KHopStructureExtractor(torch.nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree (True)
    """
    def __init__(self, embed_dim, gnn_type="gin", num_layers=2, batch_norm=True,
            concat=False, khopgnn=True):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn

        self.batch_norm = batch_norm

        self.structure_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            concat=False,
            khopgnn=True,
        )

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(2 * embed_dim)

        self.out_proj = torch.nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
            subgraph_indicator_index=None, subgraph_node_index=None,
            subgraph_edge_attr=None):

        x,x_struct = self.structure_extractor(
            origin_x = x,
            x=x[subgraph_node_index],
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            agg="sum",
        )
        x_struct = torch.cat([x, x_struct], dim=-1)
        if self.batch_norm:
            x_struct = self.bn(x_struct)
        x_struct = self.out_proj(x_struct)

        return x_struct

class StructureExtractor(torch.nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gin", num_layers=2,
                 batch_norm=True, concat=False, khopgnn=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []

        for _ in range(num_layers):
            layers.append(GINConv(embed_dim, aggr="add"))
        self.gcn = torch.nn.ModuleList(layers)
        self.relu = torch.nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(inner_dim)

        self.out_proj = torch.nn.Linear(inner_dim, embed_dim)

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, embed_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, embed_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

    def forward(self, origin_x, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        origin_x = self.x_embedding1(origin_x[:, 0]) + self.x_embedding2(origin_x[:, 1])
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)
        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return origin_x,x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x

class GCN1(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN1, self).__init__()
        self.fc=torch.nn.Linear(in_dim, out_dim)
        self.fc2=torch.nn.Linear(in_dim,out_dim)
    def forward(self,A,X):
        # A_norm=A-1e9*torch.less_equal(A,0.8)
        t = torch.triu(A, diagonal=1)
        nonzero_indices = torch.nonzero(t)
        nonzero_elements = t[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        median_value = torch.median(nonzero_elements)
        A[A < median_value] = 0
        A_norm=A.softmax(-1)
        # return 0.1*X + 0.9 * F.leaky_relu(A_norm.mm(self.fc(X)))
        return F.leaky_relu(0.1 * self.fc2(X) + 0.9 * A_norm.mm(self.fc(X)))

class GCN2(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN2, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        self.fc2 = torch.nn.Linear(in_dim, out_dim)
    def feat_aug(self,A,X):
        A_norm = A - 1e9 * torch.less_equal(A, 0.8)
        A_norm = A_norm.softmax(-1)
        return F.leaky_relu(0.1 * self.fc2(X) + 0.9 * A_norm.mm(self.fc(X)))

    def forward(self,s_emb,q_emb):
        n_support = s_emb.size(0)
        n_query = q_emb.size(0)

        s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
        q_emb_rep = q_emb.unsqueeze(1)
        s_emb_map = torch.cat((s_emb_rep, q_emb_rep), 1)

        # graph_num = s_emb_map.size()[0]
        # emb_chunk = torch.chunk(s_emb_map, graph_num, dim=0)
        # list = []
        # for i in range(graph_num):
        #     emb = emb_chunk[i].squeeze(0)
        #     x = emb / torch.norm(emb, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
        #     similarity = torch.mm(x, x.T)  # 矩阵乘法
        #     new_emb = self.feat_aug(similarity, emb)
        #     list.append(new_emb)
        # s_emb_map = torch.stack(list, dim=0)

        q,s,d = s_emb_map.shape
        emb = s_emb_map.reshape((q*s,d))
        x = emb / torch.norm(emb, dim=-1, keepdim=True)
        adj = torch.mm(x, x.T)
        new_emb = self.feat_aug(adj, emb)
        s_emb_map = new_emb.reshape((q, s, d))

        s_feat = s_emb_map[:, :-1, :]
        s_feat = s_feat[0, :, :].squeeze(0)
        q_feat = s_emb_map[:, -1, :].squeeze(1)
        return s_feat, q_feat

