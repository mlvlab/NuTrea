import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.nutrealayer import NuTreaLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class NuTrea(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init NuTrea model.
        """
        super(NuTrea, self).__init__(args, num_entity, num_relation, num_word)
        self.args = args
        self.rf_ief = args['rf_ief']
        self.layers(args)
        self.EF = 0
        self.V = 0
        self.IEF = nn.Parameter(torch.zeros((num_relation+1)*2), requires_grad=False)
        self.emb_cache = {}
        

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_layers = args['num_layers']
        self.num_expansion_ins = args['num_expansion_ins']
        self.num_backup_ins = args['num_backup_ins']
        self.lm = args['lm']
        self.post_norm = args['post_norm']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_expansion_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        for i in range(self.num_backup_ins):
            self.add_module('conreform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))
        self.freezed_bsize = nn.Parameter(torch.Tensor([args['test_batch_size']]), requires_grad=False)


    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.rf_ief :
            self.rfief_linear = nn.Linear(entity_dim, entity_dim)
        elif self.encode_type :
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())
            else :
                rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
                rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = NuTreaLayer(args, num_entity, num_relation, entity_dim)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.constraint = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            self.constraint = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'], constraint=True)
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.constraint_list, self.cons_attn_list = self.constraint(q_input, self.instruction.node_encoder)
        rel_features, rel_features_inv  = self.get_rel_feature()
        if not self.rf_ief :
            self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
            self.init_entity_emb = self.local_entity_emb
        else :
            self.local_entity_emb = None
        self.curr_dist = curr_dist
        self.dist_history = []
        self.score_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities,
                                   init_dist=curr_dist)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_ief(self, batch):
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        
        
        bsize, bnode_num = answer_dist.shape
        node_num = bsize * bnode_num
        rtype_num = self.reasoning.rel_features.shape[0]

        h2r_idx = self.reasoning.head2fact_mat.coalesce().indices()
        h2r_idx[0] = self.reasoning.batch_rels
        h2r_RF = torch.sparse_coo_tensor(h2r_idx, torch.ones_like(h2r_idx[0]), (rtype_num*2, node_num))
        
        t2r_idx = self.reasoning.tail2fact_mat.coalesce().indices()
        t2r_idx[0] = self.reasoning.batch_rels + rtype_num
        t2r_RF = torch.sparse_coo_tensor(t2r_idx, torch.ones_like(t2r_idx[0]), (rtype_num*2, node_num))

        RF = (h2r_RF + t2r_RF).coalesce()
        R2E = torch.sparse_coo_tensor(RF.indices(), torch.ones_like(RF.values()), RF.shape).coalesce()
        EF = torch.sparse.sum(R2E, 1)
        self.EF += EF.to_dense()
        self.V += self.reasoning.local_entity_mask.sum() + bsize
        IEF = torch.log(self.V / self.EF.clip(1))
        self.IEF.data = IEF.clip(0)
            
            
        
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()

        """
        Expansion and Backup Instruction Generation
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_expansion_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        self.constraint.init_reason(q_input)
        for i in range(self.num_backup_ins):
            relational_ins, attn_weight = self.constraint.get_instruction(self.constraint.relational_ins, step=i) 
            self.constraint.instructions.append(relational_ins.unsqueeze(1))
            self.constraint.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)

        """
        RF-IEF : RF(node x relations) * log(node num / (1 + EF))
        """
        if self.rf_ief :
            # Relation Frequency
            bsize, bnode_num = answer_dist.shape
            node_num = bsize * bnode_num
            rtype_num = self.reasoning.rel_features.shape[0]

            h2r_idx = self.reasoning.head2fact_mat.coalesce().indices()
            h2r_idx[0] = self.reasoning.batch_rels
            h2r_RF = torch.sparse_coo_tensor(h2r_idx, torch.ones_like(h2r_idx[0]), (rtype_num*2, node_num))
            
            t2r_idx = self.reasoning.tail2fact_mat.coalesce().indices()
            t2r_idx[0] = self.reasoning.batch_rels + rtype_num
            t2r_RF = torch.sparse_coo_tensor(t2r_idx, torch.ones_like(t2r_idx[0]), (rtype_num*2, node_num))

            RF = (h2r_RF + t2r_RF).coalesce()

            # pre computed IEF
            IEF = self.IEF.data
            IEF_coo = torch.sparse_coo_tensor(torch.arange(IEF.shape[0], device=IEF.device).tile(2,1), IEF)
            
            RFIEF = torch.sparse.mm(IEF_coo, RF.float()).coalesce()
            # if not self.rf_ief_normalize :
            #     RFIEF = torch.sparse.mm(IEF_coo, RF.float()).coalesce()
            # else :
            #     rf_idx, rf_val = RF.indices(), RF.values()
            #     for bidx in range(bsize):
            #         idx = (rf_idx[1]>bidx*bnode_num)*(rf_idx[1]<(bidx+1)*bnode_num)
            #         smpl_idx = rf_idx[:, idx]
            #         smpl_rf_val = rf_val[idx]
                    
            #         # RF
            #         smpl_RF = torch.sparse_coo_tensor(smpl_idx, smpl_rf_val, RF.shape).coalesce()
            #         smpl_RF = smpl_RF / smpl_RF.values().sum()
                    
            #         smpl_RFIEF = torch.sparse.mm(IEF_coo, smpl_RF)
                    
            #         if bidx == 0 :
            #             RFIEF = smpl_RFIEF
            #         else :
            #             RFIEF += smpl_RFIEF
                    
            #     RFIEF = RFIEF.coalesce()
                            
                
            """
            RF-IEF node init
            """
            rel_features = torch.cat([self.reasoning.rel_features, self.reasoning.rel_features_inv])
            rel_features = self.rfief_linear(rel_features)
            node_init = torch.sparse.mm(RFIEF.transpose(1,0), rel_features.to_sparse()).to_dense()
            node_init = node_init.reshape(local_entity.shape[0], local_entity.shape[1], -1)

            self.local_entity_emb = node_init
            self.init_entity_emb = node_init
            self.reasoning.local_entity_emb = node_init

        """
        NuTrea reasoning
        """
        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            relation_con = torch.cat(self.constraint.instructions, dim=1)
            self.curr_dist = current_dist            
            for j in range(self.num_layers):
                # raw_score, self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, relation_con, step=j)
                raw_score, self.curr_dist, global_rep, gs, hs = self.reasoning(self.curr_dist, relation_ins, relation_con, step=j, return_score=True)
            self.dist_history.append(self.curr_dist)
            self.score_history.append(raw_score)

            """
            Expansion Instruction Updates
            """
            for j in range(self.num_expansion_ins):
                exp_reform = getattr(self, 'reform' + str(j))
                q = exp_reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                self.instruction.instructions[j] = q.unsqueeze(1)
        
            """
            Backup Instruction Updates
            """
            for j in range(self.num_backup_ins):
                bak_reform = getattr(self, 'conreform' + str(j))
                q = bak_reform(self.constraint.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                self.constraint.instructions[j] = q.unsqueeze(1)

        
        """
        Answer Predictions
        """
        if self.post_norm :
            pred_logit = sum(self.score_history) + (1-self.reasoning.local_entity_mask) * VERY_NEG_NUMBER
            pred_dist = self.softmax_d1(pred_logit)
        else :
            pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = 0
        # for pred_dist in self.dist_history:
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        
        if self.post_norm :
            pass
        else :
            pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

    