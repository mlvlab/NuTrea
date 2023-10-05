
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as pyg
import copy

from torch_geometric.nn import MessagePassing
from .base_gnn import BaseGNNLayer


VERY_NEG_NUMBER = -100000000000

class SubgraphPool(MessagePassing):
    def __init__(self, aggr='max'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        return self.propagate(edge_index.coalesce(), x=x)


class NuTreaLayer(BaseGNNLayer):
    """
    NuTreaLayer Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(NuTreaLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_expansion_ins = args['num_expansion_ins']
        self.num_backup_ins = args['num_backup_ins']
        self.num_layers = args['num_layers']
        self.context_coef = args['context_coef']
        self.backup_depth = args['backup_depth']
        self.agg = 'max'
        self.post_norm = args['post_norm']
        self.use_posemb = args['pos_emb']

        self.init_layers(args)


    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.sigmoid = nn.Sigmoid()
        self.softmax_d1 = nn.Softmax(dim=1)
        self.g_score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.h_score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        self.aggregator = SubgraphPool(aggr=self.agg)
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_layers):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('con_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*self.num_expansion_ins*entity_dim + entity_dim, out_features=entity_dim))
            self.add_module('s2e_linear' + str(i), nn.Linear(in_features=2*self.num_backup_ins*entity_dim + entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear2' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('s2e_linear2' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
            self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))

        self.lin_m =  nn.Linear(in_features=(self.num_expansion_ins)*entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, init_dist, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities
        self.leaf_nodes = init_dist.detach()
       

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb, inverse=False):
        """
        Aggregates neighbor representations
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv if inverse else self.rel_features
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)  # total edge num x D
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)  # total edge num x D
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
            
        if inverse :
            fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
            fact_val = fact_val * fact_prior
            f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        else :
            fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))
            fact_val = fact_val * fact_prior
            f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)

        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def set_batch_adj(self):
        adj = self.head2tail_mat.transpose(1,0).coalesce()
        idx, vals = pyg.utils.remove_self_loops(adj.indices(), adj.values())
        self.adj = torch.sparse_coo_tensor(idx, vals, adj.size()).coalesce()
        
        adj_inv = self.head2tail_mat.coalesce()
        idx, vals = pyg.utils.remove_self_loops(adj_inv.indices(), adj_inv.values())
        self.adj_inv = torch.sparse_coo_tensor(idx, vals, adj_inv.size()).coalesce()


    def get_next_leaf(self, leaf_nodes, inverse=False):
        adj = self.adj if not inverse else self.adj_inv
        x = torch.sparse.mm(adj, leaf_nodes.view(-1, 1)).view(leaf_nodes.shape)
        return (x > 0.0).float()
    

    def pool_subgraph(self, leaf_nodes, constrainer, con_linear, depth=1, inverse=False):

        batch_size = self.batch_size
        max_local_entity = self.max_local_entity

        # leaf node acculation
        leaf_nodes = self.get_next_leaf(leaf_nodes, inverse=inverse)
        leaf_nodes_list = [leaf_nodes]
        for _ in range(depth-1) :
            leaf_nodes = self.get_next_leaf(leaf_nodes, inverse=inverse)
            leaf_nodes_list.append((leaf_nodes_list[-1] + leaf_nodes > 0.0).float())
        leaf_nodes_list.reverse()

        # query for constraints
        rel_features = self.rel_features_inv if inverse else self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        con_query = torch.index_select(constrainer, dim=0, index=self.batch_ids)
        con_val = F.relu(con_linear(fact_rel) * con_query)

        # neighbor relation aggregation
        fact2node_mat = self.fact2tail_mat if inverse else self.fact2head_mat
        con_pooled = self.aggregator(con_val, fact2node_mat)
        con_pooled = leaf_nodes_list[0].flatten().unsqueeze(-1) * con_pooled 

        if depth > 1 :
            adj = self.head2tail_mat.transpose(1,0).coalesce() if inverse else self.head2tail_mat.coalesce()
            idx, vals = pyg.utils.add_remaining_self_loops(adj.indices(), adj.values())
            adj = torch.sparse_coo_tensor(idx, vals, adj.size()).coalesce()
            for d in range(depth-1) :
                con_pooled = self.aggregator(con_pooled, adj)
                con_pooled = leaf_nodes_list[d+1].flatten().unsqueeze(-1) * con_pooled 
        
        pooled_rep = con_pooled.view(batch_size, max_local_entity, self.entity_dim)

        return pooled_rep


    def forward(self, current_dist, relational_ins, relational_con, step=0, return_score=False):
        
        """
        Compute next probabilistic vectors and current node representations.
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        con_linear = getattr(self, 'con_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        s2e_linear = getattr(self, 's2e_linear' + str(step))
        e2e_linear2 = getattr(self, 'e2e_linear2' + str(step))
        s2e_linear2 = getattr(self, 's2e_linear2' + str(step))
        if self.use_posemb :
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else :
            pos_emb, pos_emb_inv = None, None
        # score_func = getattr(self, 'score_func' + str(step))
        expansion_score_func = self.g_score_func
        backup_score_func = self.h_score_func

        self.set_batch_adj()
        

        """
        Expansion
        """
        neighbor_reps = []
        
        for j in range(relational_ins.size(1)):
                
            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb) # B x 2000 x D
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb_inv, inverse=True)
            neighbor_reps.append(neighbor_rep)

        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        self.local_entity_emb = e2e_linear2(F.relu(e2e_linear(self.linear_drop(next_local_entity_emb))))

        expansion_score = expansion_score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)


        """
        Backup
        """
        subgraph_reps = []
        for j in range(relational_con.size(1)):
            pooled_rep = self.pool_subgraph(self.leaf_nodes, relational_con[:,j,:], con_linear, depth=self.backup_depth)
            subgraph_reps.append(pooled_rep)

            pooled_rep = self.pool_subgraph(self.leaf_nodes, relational_con[:,j,:], con_linear, depth=self.backup_depth, inverse=True)
            subgraph_reps.append(pooled_rep)
            
        subgraph_reps = torch.cat(subgraph_reps, dim=2)
        next_local_entity_emb = torch.cat((self.local_entity_emb, subgraph_reps), dim=2)
        self.local_entity_emb = s2e_linear2(F.relu(s2e_linear(self.linear_drop(next_local_entity_emb))))

        backup_score = backup_score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)

        score_tp = expansion_score + self.context_coef * backup_score


        """
        Distribution Update
        """
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        prenorm_score = score_tp if self.post_norm else None
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return prenorm_score, current_dist, self.local_entity_emb, \
                expansion_score + (1 - answer_mask) * VERY_NEG_NUMBER, backup_score + (1 - answer_mask) * VERY_NEG_NUMBER
        
        
        return prenorm_score, current_dist, self.local_entity_emb


