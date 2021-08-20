#coding:utf-8
import tensorflow as tf
from collections import defaultdict

class BiKBGAT(object):
    def __init__(self,params,entity_embedding,relation_embedding):
        self.optimizer=params.optimizer
        self.init_params(params,entity_embedding,relation_embedding)
        self.name=self.__class__.__name__

        self.features=self.build_placeholder()
        self.build_model(self.features)
    def init_params(self,params,entity_embedding,relation_embedding):
        self.init_entity_embedding=entity_embedding
        self.init_relation_embedding=relation_embedding
        self.entity_embedding_weights=None
        self.relation_embedding_weights = None
        self.margin=params.margin or 1

        #kb
        self.triple_num=params.triple_num
        self.init_emb_dim=params.init_emb_dim
        self.use_nhop=params.use_nhop

        #model size
        self.entity_vocab_size=params.entity_vocab_size
        self.relation_vocab_size=params.relation_vocab_size
        self.emb_dim=params.emb_dim
        self.init_emb_dim=params.init_emb_dim
        self.hidden_dim=params.hidden_dim
        self.att_hidden_dim=params.att_hidden_dim
        self.num_filter=params.num_filter or 500
        self.num_attention_head=params.num_attention_head or 2
        self.initializer=params.initializer
        self.leaky_relu_alph=params.leaky_relu_alph or 0.2
        if self.init_entity_embedding is not None:
            assert self.init_entity_embedding.shape[1]==self.init_emb_dim,(self.init_entity_embedding.shape,self.init_emb_dim)
        #train
        self.num_neg=params.num_neg
        self.margin=params.margin or 0.1
        self.margin_transE=params.margin_transE or 5
        self.activation=params.activation or tf.nn.relu
        self.num_layer=params.num_layer or 0
        self.use_relation_attention=params.use_relation_attention or False
        # self.lr=params.lr
        self.l2_reg_lambda=params.l2_reg_lambda or 0
        self.l1_reg_lambda=params.l2_reg_lambda or 0.001
        self.label_smooth_epsilon=params.label_smooth_epsilon or 0.1
        self.optimizer=params.optimizer
        self.init_checkpoint=params.init_checkpoint

        #loss
        self.gamma=params.gamma or 5

    def build_model(self,features):
        '''构建模型'''
        scores,scores_reverse=self.forward(features,reuse=False)
        self.scores = scores
        self.scores_reverse=scores_reverse

        #decode
        scores_prediect,scores_prediect_reverse=self.decode(features,reuse=True)
        self.scores_predict=scores_prediect
        self.scores_predict_reverse=scores_prediect_reverse

        #labels
        # labels=features["labels"]
        labels=features["inputs"]["tail"]
        labels_head=features["inputs"]["head"]
        label_masks=features["label_masks"]
        label_masks_reverse=features["label_masks_reverse"]

        self.entity_embeddings=self.entity_embedding_weights
        self.relation_embeddings=self.relation_embedding_weights

        #compute loss
        self.loss=self.compute_loss_softmax(scores,labels,labels_mask=None)
        self.loss_reverse=self.compute_loss_softmax(scores_reverse,labels_head,labels_mask=None)
        self.loss=self.loss+self.loss_reverse

        # self.loss=self.compute_loss_hinge(scores,labels,labels_mask=label_masks)
        # self.loss_reverse=self.compute_loss_hinge(scores_reverse,labels_head,labels_mask=label_masks_reverse)
        # self.loss=self.loss+self.loss_reverse

        # self.loss=self.compute_loss_softmargin(scores,labels)
        # self.loss_reverse=self.compute_loss_softmargin(scores,labels_head)
        # self.loss=self.loss+self.loss_reverse

        # self.loss=self.compute_loss_cross_entropy(scores,labels)
        # self.loss_reverse=self.compute_loss_cross_entropy(scores,labels_head)
        # self.loss=self.loss+self.loss_reverse
        if self.l2_reg_lambda:
            #l2 loss
            l2_loss=tf.add_n(tf.get_collection("l2"))
            self.loss=self.loss+l2_loss*self.l2_reg_lambda

        self.right_num,self.acc=self.compute_acc(scores,labels)
        self.train_op=self.get_train_op(self.loss,learning_rate=self.lr)
        self.predict_outputs={"output":self.scores,"loss":self.loss,"acc":self.acc} #预测时需要获取的值
        self.train_outputs={"train_op":self.train_op,"loss":self.loss,"acc":self.acc,"output":self.scores} #训练时需要获取的值
    def gat(self,features):
        heads=features["adjacency"]["head"]
        tails=features["adjacency"]["tail"]
        relations=features["adjacency"]["relation"]
        nhop_heads=features["nhop"]["head"]
        nhop_tails=features["nhop"]["tail"]
        nhop_relations=features["nhop"]["relation"]

        with tf.variable_scope("embdding"):
            if self.init_entity_embedding is not None:
                print("init embedding from pre-trained")
                entity_embeddings=tf.get_variable(name="entity_embeddings", initializer=self.init_entity_embedding,
                                                                  dtype=tf.float32)
                relation_embeddings= tf.get_variable(name="relation_embeddings", initializer=self.init_relation_embedding,
                                                              dtype=tf.float32)
                relation_embeddings_reverse= tf.get_variable(name="relation_embeddings_reverse", initializer=self.init_relation_embedding,
                                                              dtype=tf.float32)
            else:
                entity_embeddings=tf.get_variable(name="entity_embeddings", shape=[self.entity_vocab_size,self.emb_dim],
                                                                  dtype=tf.float32)
                relation_embeddings= tf.get_variable(name="relation_embeddings",shape=[self.entity_vocab_size,self.emb_dim],
                                                              dtype=tf.float32)
                relation_embeddings_reverse= tf.get_variable(name="relation_embeddings_reverse",shape=[self.entity_vocab_size,self.emb_dim],
                                                              dtype=tf.float32)

        relation_emb=relation_embeddings
        relation_emb_reverse=relation_embeddings_reverse
        entity_emb=entity_embeddings
        att_heads=self.num_attention_head
        triple_masks=features["triple_mask"]
        num_layer=self.num_layer
        entity_outputs=[tf.expand_dims(entity_embeddings,axis=1)]
        relation_outputs=[relation_embeddings]
        relation_outputs_reverse=[relation_embeddings_reverse]
        for i in range(num_layer):
            with tf.variable_scope("layer_%s"%i):
                entity_emb=self.multiHeadGAT(heads,relations,tails,nhop_heads,nhop_relations,nhop_tails,
                                           in_dim=self.init_emb_dim,rel_dim=self.init_emb_dim,out_dim=self.hidden_dim,
                                           entity_embeddings=entity_emb,relation_embeddings=relation_emb,relation_embeddings_reverse=relation_emb_reverse,
                                            num_att_heads=att_heads,use_layer_norm=False,use_resdual=True)
            with tf.variable_scope("relation_embedding_%s"%i):
                W=tf.get_variable(name="weights",shape=[self.init_emb_dim,self.hidden_dim])
                # tf.add_to_collection("l2", tf.nn.l2_loss(W))
                # relation_emb=tf.nn.relu(tf.matmul(relation_emb,W))
                relation_emb=tf.nn.relu(tf.matmul(relation_emb,W))+relation_emb
                # relation_emb=tf.layers.batch_normalization(relation_emb,axis=-1)
                #relation reverse
                # relation_emb_reverse=tf.nn.relu(tf.matmul(relation_emb_reverse,W))
                relation_emb_reverse=tf.nn.relu(tf.matmul(relation_emb_reverse,W))+relation_emb_reverse
                # relation_emb=tf.layers.batch_normalization(relation_emb,axis=-1)

            entity_outputs.append(tf.expand_dims(entity_emb,axis=1))
            relation_outputs.append(relation_emb)
            relation_outputs_reverse.append(relation_emb_reverse)

        #attention
        with tf.variable_scope("attention_outputs"):
            entity_outputs=tf.concat(entity_outputs,axis=1)
            hop_embedding=tf.get_variable(name="hop_emb",shape=[num_layer+1,self.hidden_dim],initializer=tf.initializers.zeros)
            hop_embedding=tf.expand_dims(hop_embedding,axis=0)
            entity_outputs_hop=entity_outputs+hop_embedding
            W=tf.get_variable(name="weights",shape=[self.hidden_dim,1])
            scores=tf.matmul(tf.reshape(entity_outputs_hop,shape=[-1,self.hidden_dim]),W)
            scores=tf.reshape(scores,shape=[-1,num_layer+1])
            scores=tf.nn.softmax(scores)
            scores=tf.expand_dims(scores,axis=2)
            entity_emb_new=tf.reduce_sum(entity_outputs_hop*scores,axis=1)
        if self.use_relation_attention:
            with tf.variable_scope("relation_attention_outputs"):
                relation_outputs=[tf.expand_dims(rel_emb,1) for rel_emb in relation_outputs]
                relation_outputs=tf.concat(relation_outputs,axis=1)
                # hop_embedding=tf.get_variable(name="hop_emb",shape=[num_layer+1,self.hidden_dim],initializer=tf.initializers.zeros)
                # hop_embedding=tf.expand_dims(hop_embedding,axis=0)
                relation_outputs_hop=relation_outputs+hop_embedding
                W=tf.get_variable(name="weights",shape=[self.hidden_dim,1])
                scores=tf.matmul(tf.reshape(relation_outputs_hop,shape=[-1,self.hidden_dim]),W)
                scores=tf.reshape(scores,shape=[-1,num_layer+1])
                scores=tf.nn.softmax(scores)
                scores=tf.expand_dims(scores,axis=2)
                relation_emb_new=tf.reduce_sum(relation_outputs_hop*scores,axis=1)
                #relation reverse
                relation_outputs_reverse=[tf.expand_dims(rel_emb,1) for rel_emb in relation_outputs_reverse]
                relation_outputs_reverse=tf.concat(relation_outputs_reverse,axis=1)
                relation_outputs_reverse_hop=relation_outputs_reverse+hop_embedding
                scores=tf.matmul(tf.reshape(relation_outputs_reverse_hop,shape=[-1,self.hidden_dim]),W)
                scores=tf.reshape(scores,shape=[-1,num_layer+1])
                scores=tf.nn.softmax(scores)
                scores=tf.expand_dims(scores,axis=2)
                relation_emb_reverse_new=tf.reduce_sum(relation_outputs_reverse_hop*scores,axis=1)
        else:
            relation_emb_new=relation_outputs[-1]
            relation_emb_reverse_new=relation_outputs_reverse[-1]


        return entity_emb_new,relation_emb_new,relation_emb_reverse_new

    def forward(self,features,reuse=False):
        entity_embeddings,relation_embeddings,relation_embeddings_reverse=self.gat(features)
        self.final_entity_embeddings=entity_embeddings
        self.final_relation_embeddings=relation_embeddings
        self.final_relation_embeddings_reverse=relation_embeddings_reverse
        heads=features["inputs"]["head"]
        tails=features["inputs"]["tail"]
        relations=features["inputs"]["relation"]
        #embed
        head_emb=tf.nn.embedding_lookup(entity_embeddings,heads)
        tail_emb=tf.nn.embedding_lookup(entity_embeddings,tails)
        relation_emb=tf.nn.embedding_lookup(relation_embeddings,relations)
        relation_emb_reverse=tf.nn.embedding_lookup(relation_embeddings_reverse,relations)
        # # l2
        # tf.add_to_collection("l2",tf.nn.l2_loss(head_emb))
        # tf.add_to_collection("l2",tf.nn.l2_loss(relation_emb))
        # tf.add_to_collection("l2",tf.nn.l2_loss(relation_emb_reverse))
        # tf.add_to_collection("l2",tf.nn.l2_loss(tail_emb))
        #TransE
        # head_emb=tf.nn.l2_normalize(head_emb,-1)
        # tail_emb=tf.nn.l2_normalize(tail_emb,-1)
        # relation_emb=tf.nn.l2_normalize(relation_emb,-1)
        # scores=abs(head_emb+relation_emb-tail_emb)
        # scores=tf.reduce_sum(scores,axis=-1,keepdims=True)

        #DistMult
        # scores=head_emb*relation_emb*tail_emb
        # scores=tf.reduce_sum(scores,axis=-1,keep_dims=True)
        #Dense
        scores,scores_reverse=self.get_scores(head_emb,relation_emb,tail_emb,relation_emb_reverse,entity_embeddings,reuse=reuse)
        return scores,scores_reverse

    def decode(self,features,reuse=True):
        entity_embeddings=features["entity_embedding"]
        relation_embeddings=features["relation_embedding"]
        relation_embeddings_reverse=features["relation_embedding_reverse"]
        heads=features["inputs"]["head"]
        tails=features["inputs"]["tail"]
        relations=features["inputs"]["relation"]
        #embed
        head_emb=tf.nn.embedding_lookup(entity_embeddings,heads)
        tail_emb=tf.nn.embedding_lookup(entity_embeddings,tails)
        relation_emb=tf.nn.embedding_lookup(relation_embeddings,relations)
        relation_emb_reverse=tf.nn.embedding_lookup(relation_embeddings_reverse,relations)
        #TransE
        # head_emb=tf.nn.l2_normalize(head_emb,-1)
        # tail_emb=tf.nn.l2_normalize(tail_emb,-1)
        # relation_emb=tf.nn.l2_normalize(relation_emb,-1)
        # scores=abs(head_emb+relation_emb-tail_emb)
        # scores=tf.reduce_sum(scores,axis=-1,keepdims=True)
        #DistMult
        # scores=head_emb*relation_emb*tail_emb
        # scores=tf.reduce_sum(scores,axis=-1,keep_dims=True)
        #Dense
        scores,scores_reverse=self.get_scores(head_emb,relation_emb,tail_emb,relation_emb_reverse,entity_embeddings,reuse=reuse)
        return scores, scores_reverse

    def get_scores(self,head_emb,relation_emb,tail_emb,relation_emb_reverse,entity_embeddings,reuse=False):
        # Dense
        with tf.variable_scope("target", reuse=reuse):
            # entity_embeddings_target=self.init_entity_embedding
            # entity_embeddings_target = tf.get_variable(name="weights", initializer=self.init_entity_embedding)
            W = tf.get_variable(name="weights1", shape=[self.hidden_dim, self.hidden_dim * 10],
                                initializer=tf.initializers.he_normal())
            tf.add_to_collection("l2", tf.nn.l2_loss(W))
            b = tf.get_variable(name="bias", shape=[self.hidden_dim * 10])
            entity_embeddings_target = tf.matmul(entity_embeddings, W) + b
            entity_embeddings_target = tf.nn.relu(entity_embeddings_target)
            #FB15k-237加dropout, WN18RR不加
            entity_embeddings_target=tf.nn.dropout(entity_embeddings_target,keep_prob=self.keep_prob)

        with tf.variable_scope("scores", reuse=reuse):
            W = tf.get_variable(name="weights", shape=[self.hidden_dim * 2, self.hidden_dim*10],
                                initializer=tf.initializers.he_normal())
            tf.add_to_collection("l2", tf.nn.l2_loss(W))
            b = tf.get_variable(name="bias", shape=[self.hidden_dim*10])
            W2 = tf.get_variable(name="weights2", shape=[self.hidden_dim * 10, self.hidden_dim],
                                 initializer=tf.initializers.he_normal())
            # tf.add_to_collection("l2",tf.nn.l2_loss(W))
            b2 = tf.get_variable(name="bias2", shape=[self.hidden_dim])
            # query
            query = tf.concat([head_emb, relation_emb], axis=-1)
            query = tf.matmul(query, W) + b
            query = tf.nn.relu(query)
            query = tf.nn.dropout(query, keep_prob=self.keep_prob)
            # query=tf.matmul(query,W2)+b2
            # query=tf.nn.relu(query)
            # scores
            scores = tf.matmul(query, entity_embeddings_target, transpose_b=True)
            # query reverse
            query_reverse = tf.concat([tail_emb, relation_emb_reverse], axis=-1)
            query_reverse = tf.matmul(query_reverse, W) + b
            query_reverse = tf.nn.relu(query_reverse)
            query_reverse = tf.nn.dropout(query_reverse, keep_prob=self.keep_prob)
            # query_reverse=tf.matmul(query_reverse,W2)+b2
            # query_reverse=tf.nn.relu(query_reverse)
            scores_reverse = tf.matmul(query_reverse, entity_embeddings_target, transpose_b=True)

            return scores,scores_reverse


    def build_placeholder(self):
        features=defaultdict(lambda :defaultdict(dict))
        #final embedding
        features["entity_embedding"]=tf.placeholder(dtype=tf.float32,shape=[self.entity_vocab_size,self.hidden_dim])
        features["relation_embedding"]=tf.placeholder(dtype=tf.float32,shape=[self.relation_vocab_size,self.hidden_dim])
        features["relation_embedding_reverse"]=tf.placeholder(dtype=tf.float32,shape=[self.relation_vocab_size,self.hidden_dim])
        #输入数据
        features["inputs"]["head"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["inputs"]["tail"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["inputs"]["relation"]=tf.placeholder(dtype=tf.int32,shape=[None])
        #adjacency matrix 邻接矩阵
        features["adjacency"]["head"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["adjacency"]["tail"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["adjacency"]["relation"]=tf.placeholder(dtype=tf.int32,shape=[None])
        # features["adjacency"]["neighbor_indices"]=tf.placeholder(dtype=tf.int32,shape=[None,2]) #用来构建entity_num*triple_num的稀疏矩阵
        #nhop adjacency matrix
        features["nhop"]["head"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["nhop"]["tail"]=tf.placeholder(dtype=tf.int32,shape=[None])
        features["nhop"]["relation"]=tf.placeholder(dtype=tf.int32,shape=[None,None])
        features["labels"]=tf.placeholder(dtype=tf.float32,shape=[None])
        features["label_masks"]=tf.placeholder(dtype=tf.float32,shape=[None,self.entity_vocab_size])
        features["label_masks_reverse"]=tf.placeholder(dtype=tf.float32,shape=[None,self.entity_vocab_size])
        features["triple_mask"]=tf.placeholder(dtype=tf.int64,shape=[None])
        self.keep_prob=features["keep_prob"]=tf.placeholder(dtype=tf.float32,name="keep_prob")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.lr=tf.placeholder(dtype=tf.float32,name="keep_prob")
        return features
    def multiHeadGAT(self,heads,relations,tails,nhop_heads,nhop_relations,nhop_tails,
                       in_dim,rel_dim,out_dim,entity_embeddings,relation_embeddings,relation_embeddings_reverse,
                        num_att_heads=2,use_layer_norm=False,use_resdual=True):
        outputs=[]
        for i in range(num_att_heads):
            with tf.variable_scope("att_head_%s"%i):
                output=self.sparseBiGATLayer(heads,relations,tails,nhop_heads,nhop_relations,nhop_tails,
                       in_dim,rel_dim,out_dim,entity_embeddings,relation_embeddings,relation_embeddings_reverse)
                outputs.append(tf.expand_dims(output,1))
        # outputs=outputs_in+outputs_out
        outputs=tf.concat(outputs,axis=1)
        outputs=tf.reduce_mean(outputs,axis=1)
        outputs=tf.reshape(outputs,[self.entity_vocab_size,self.hidden_dim])

        # outputs=tf.nn.relu(outputs)

        if use_resdual:
            outputs=outputs+entity_embeddings
            # outputs=tf.layers.batch_normalization(outputs,axis=-1)
        if use_layer_norm:
            # outputs=layer_norm(outputs)
            outputs=tf.nn.l2_normalize(outputs)
        return outputs

    def sparseBiGATLayer(self,heads,relations,tails,nhop_heads,nhop_relations,nhop_tails,
                       in_dim,rel_dim,out_dim,entity_embeddings,relation_embeddings,relation_embeddings_reverse):
        n_e=self.entity_vocab_size
        n_r=self.relation_vocab_size
        #embedding
        head_emb=tf.nn.embedding_lookup(entity_embeddings,heads)
        relation_emb=tf.nn.embedding_lookup(relation_embeddings,relations)
        relation_emb_reverse=tf.nn.embedding_lookup(relation_embeddings_reverse,relations)

        tail_emb=tf.nn.embedding_lookup(entity_embeddings,tails)
        triples=tf.concat([head_emb,relation_emb,tail_emb],axis=-1)
        triples_reverse=tf.concat([head_emb,relation_emb_reverse,tail_emb],axis=-1)
        #nhop triples
        if self.use_nhop:
            print("use nhop neighbors....")
            #nhop embedding
            nhop_head_emb=tf.nn.embedding_lookup(entity_embeddings,nhop_heads)
            nhop_tail_emb=tf.nn.embedding_lookup(entity_embeddings,nhop_tails)
            nhop_relation_emb=tf.nn.embedding_lookup(relation_embeddings,nhop_relations)
            nhop_relation_emb=tf.reduce_sum(nhop_relation_emb,axis=1,keepdims=False)
            nhop_relation_emb_reverse=tf.nn.embedding_lookup(relation_embeddings_reverse,nhop_relations)
            nhop_relation_emb_reverse=tf.reduce_sum(nhop_relation_emb_reverse,axis=1,keepdims=False)

            nhop_triples=tf.concat([nhop_head_emb,nhop_relation_emb,nhop_tail_emb],axis=-1)
            nhop_triples_reverse=tf.concat([nhop_head_emb,nhop_relation_emb_reverse,nhop_tail_emb],axis=-1)
            #concat one-hop and nhop
            triples=tf.concat([triples,nhop_triples],axis=0)
            triples_reverse=tf.concat([triples_reverse,nhop_triples_reverse],axis=0)
            tails=tf.concat([tails,nhop_tails],axis=0)
            heads=tf.concat([heads,nhop_heads],axis=0)
        triple_num=tf.shape(triples,out_type=tf.int32)[0]
        #每个节点的邻居三元组位置[node_id,triple_id]
        indices_in = tf.concat([tf.expand_dims(tails, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        indices_out = tf.concat([tf.expand_dims(heads, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        with tf.variable_scope("in_direction_gat"):
            with tf.variable_scope("gat_triple_dense"):
                W=tf.get_variable(name="weights",shape=[in_dim*2+rel_dim,out_dim],initializer=tf.initializers.he_normal())
                tf.add_to_collection("l2",tf.nn.l2_loss(W))
            triples_in=tf.matmul(triples,W)
            #attention score
            with tf.variable_scope("gat_att"):
                W=tf.get_variable(name="weights",shape=[out_dim,1],initializer=tf.initializers.he_normal())
                # tf.add_to_collection("l2", tf.nn.l2_loss(W))
                scores=tf.matmul(triples_in,W)
                scores=tf.nn.leaky_relu(scores)
                scores=tf.exp(scores)
                # scores=scores*triple_masks
                scores=tf.nn.dropout(scores,keep_prob=self.keep_prob)
            #score Sparse Matrix
            indices=tf.cast(indices_in,tf.int64)
            scores=tf.squeeze(scores,1)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_in=tf.sparse_tensor_dense_matmul(scores_matrix,triples_in)/(scores_sum+1e-8)
        with tf.variable_scope("out_direction_gat"):
            with tf.variable_scope("gat_triple_dense"):
                W=tf.get_variable(name="weights",shape=[in_dim*2+rel_dim,out_dim],initializer=tf.initializers.he_normal())
                tf.add_to_collection("l2", tf.nn.l2_loss(W))
            triples_out=tf.matmul(triples_reverse,W)
            #attention score
            with tf.variable_scope("gat_att"):
                W=tf.get_variable(name="weights",shape=[out_dim,1],initializer=tf.initializers.he_normal())
                # tf.add_to_collection("l2", tf.nn.l2_loss(W))
                scores=tf.matmul(triples_out,W)
                scores=tf.nn.leaky_relu(scores)
                scores=tf.exp(scores)
                # scores=scores*triple_masks
                scores=tf.nn.dropout(scores,keep_prob=self.keep_prob)
            #score Sparse Matrix
            indices=tf.cast(indices_out,tf.int64)
            scores=tf.squeeze(scores,1)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_out=tf.sparse_tensor_dense_matmul(scores_matrix,triples_out)/(scores_sum+1e-8)
        # return outputs_in,outputs_out
        #merge in and out direction
        with tf.variable_scope("outputs"):
            outputs=tf.concat([outputs_in,outputs_out],axis=-1)
            outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)
            outputs=tf.nn.relu(outputs)
            W=tf.get_variable(name="weights",shape=[out_dim*2,out_dim],initializer=tf.initializers.he_normal())
            tf.add_to_collection("l2",tf.nn.l2_loss(W))
            outputs=tf.matmul(outputs,W)
            # outputs=tf.nn.relu(outputs)
        return outputs

    def compute_loss(self,scores_pos,scores_neg):
        losses=tf.maximum(1e-8,self.margin+scores_pos-scores_neg)
        loss=tf.reduce_mean(losses,axis=-1)
        loss=tf.reduce_mean(loss)
        return loss
    def compute_loss_hinge(self,scores,labels,labels_mask=None):
        labels_onehot=tf.one_hot(labels,depth=self.entity_vocab_size,dtype=tf.float32)
        scores_pos=tf.reduce_sum(scores*labels_onehot,axis=-1,keep_dims=True)

        losses=tf.maximum(1e-8,self.margin+scores-scores_pos)
        if labels_mask is not None:
            losses=losses*(1-labels_mask)

        loss=tf.reduce_mean(losses,axis=-1)
        loss=tf.reduce_mean(loss)
        return loss

    def compute_loss_softmargin(self,scores,labels):
        labels_onehot = tf.one_hot(labels, depth=self.entity_vocab_size, dtype=tf.float32)
        labels=-1*(1-labels_onehot)+labels_onehot
        # scores=tf.squeeze(scores,axis=1)
        losses=tf.nn.softplus(-1*scores*labels)
        loss=tf.reduce_sum(losses,axis=1)
        loss=tf.reduce_mean(loss)
        return loss
    def compute_loss_cross_entropy(self,scores,labels):
        labels_onehot = tf.one_hot(labels, depth=self.entity_vocab_size, dtype=tf.float32)
        # scores=tf.squeeze(scores,axis=1)
        scores=tf.clip_by_value(tf.nn.sigmoid(scores),clip_value_min=1e-8,clip_value_max=1-1e-8)
        losses=labels_onehot*tf.log(scores)+(1-labels_onehot)*tf.log(1-scores)
        loss=tf.reduce_sum(losses,axis=1)
        loss=tf.reduce_mean(loss)
        return loss

    def compute_loss_softmax(self,scores,labels,labels_mask=None):
        scores=self.gamma*scores
        if labels_mask is not None:
            scores=scores+(-10000)*labels_mask
        # losses=tf.losses.sparse_softmax_cross_entropy(labels,scores)
        losses=tf.losses.sparse_softmax_cross_entropy(labels,scores,reduction="none")
        #加权重
        # probs=tf.exp(-losses)
        # weights=tf.pow(1-probs,2)
        # losses=losses*weights

        loss=tf.reduce_mean(losses)
        return loss

    def compute_acc(self,scores,labels):
        # rights=tf.cast(tf.greater(score_pos,score_neg),dtype=tf.float32)
        # rights=tf.cast(tf.greater(scores[:,0],tf.reduce_max(scores[:,1:],axis=1,keepdims=True)),dtype=tf.float32)
        # rights=tf.cast(tf.greater(scores[:,1],scores[:,0]),dtype=tf.float32)
        preds=tf.cast(tf.arg_max(scores,-1),dtype=tf.int32)
        rights=tf.cast(tf.equal(preds,labels),dtype=tf.float32)

        right_num=tf.reduce_sum(rights)
        acc=tf.reduce_mean(rights)
        return right_num,acc

    def get_train_op(self,loss,learning_rate=0.001):
        optimizer=self.optimizer(learning_rate)
        var_list=tf.trainable_variables()
        grad_vars=optimizer.compute_gradients(loss,var_list=var_list,aggregation_method=2)
        grad_vars=[(tf.clip_by_value(g,clip_value_min=-5,clip_value_max=5),v) for g,v in grad_vars if g is not None]
        train_op=optimizer.apply_gradients(grad_vars)
        return train_op
