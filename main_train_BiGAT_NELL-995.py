#coding:utf-8
import tensorflow as tf
from helpers.data_helper import VocabHelper,GraphDataHelperGAT,NeighborDataHelper
from models_BiGAT import BiKBGAT
from helpers.trainer import GATTrainer
from helpers.params import Params
from helpers.data_evaluate import evaluate_model

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）

data_dir="NELL-995"
kb_params = {
    #data
    "data_dir":data_dir,
    "entity_vocab_size": 75492,
    "relation_vocab_size": 200,
    "triple_num":149678,
    "use_nhop":True,
    #model
    "init_emb_dim":100,
    "emb_dim":100,
    "hidden_dim": 100,
    "num_filter": 100,
    "num_layer": 2,
    "use_relation_attention": True,
    # loss
    "margin": 5,
    "gamma": 5,
    "num_neg": 2,
    "l2_reg_lambda": 1e-6,
    # train
    "optimizer": tf.train.AdamOptimizer,
    "lr": 0.0005,
    "keep_prob": 0.5,
    "batch_size": 1024,
    # nhop
    "max_nhop_rule_num": 2,
    # test
    "skip_unique_entities": True,
    "type_constraint": False
    }


if __name__=="__main__":
    params=Params(data_dir)
    params.update(kb_params)

    vocabHelper=VocabHelper(params=params)
    neighborHelper=NeighborDataHelper(params.kb_path, vocabHelper, params)

    valid_dataHelper=GraphDataHelperGAT(dataPath=params.valid_dataPath,params=params,vocabHelper=vocabHelper,neighborHelper=neighborHelper)
    test_dataHelper = GraphDataHelperGAT(dataPath=params.test_dataPath, params=params, vocabHelper=vocabHelper,neighborHelper=neighborHelper)
    #datasets
    train_dataHelper=GraphDataHelperGAT(dataPath=params.train_dataPath,params=params,vocabHelper=vocabHelper,neighborHelper=neighborHelper,
                                        test_triples=test_dataHelper.triples,valid_triples=valid_dataHelper.triples)

    entity_embeddings=vocabHelper.entity_embeddings
    relation_embeddings=vocabHelper.relation_embeddings
    entities = list(vocabHelper.entity2id.keys())
    print(entity_embeddings.shape,len(entities))

    with tf.Session() as sess:
        #model
        model=BiKBGAT(params,entity_embedding=vocabHelper.entity_embeddings,relation_embedding=vocabHelper.relation_embeddings)
        #train
        trainer=GATTrainer(model,params)
        sess.run(tf.global_variables_initializer())
        # trainer.restore_last_session(sess)
        trainer.train(sess,data_helper=train_dataHelper,eval_data_helper=valid_dataHelper,test_data_helper=test_dataHelper,iter_num=200)
        #predict
        entity_embeddings,relation_embeddings,relation_embeddings_reverse=trainer.predict_embeddings(sess,train_dataHelper)
        #evaluate
        evaluate_model(sess,trainer,test_dataHelper)





