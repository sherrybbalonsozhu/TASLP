#coding:utf-8
import numpy as np
import tensorflow as tf
import logging
import os
import time
from helpers.data_evaluate import compute_metrics

t=time.localtime()
mon=t[1]
date=t[2]
h=t[3]
m=t[4]        
def getLogger(path,data_dir,name="Logger",mode="a"):
    logger=logging.Logger(name)
    logger.setLevel(logging.INFO)
    # name="%s-%s-%s_%s"%(mon,date,h,name)
    name="%s-%s-%s-%s"%(mon,date,name,data_dir)
    filename=os.path.join(path,name)
    fh=logging.FileHandler(filename=filename,mode=mode)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("add logger")
    return logger


class BaseTrainer(object):
    def __init__(self, model, params):
        self.model = model
        self.init_params(params)
        data_dir=params.data_dir
        name = model.__class__.__name__
        # 参数保存路径
        self.ckpt_path = os.path.join(params.weight_path, "%s_%s_%s" % (name, data_dir,self.num_layer))
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.log_path = os.path.join(params.log_path, model.__class__.__name__)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log = getLogger(path=self.log_path, data_dir=data_dir,name=model.__class__.__name__)

    def init_params(self, params):
        self.batch_size = params.batch_size
        self.best_acc = 0.
        self.best_loss = np.inf
        self.global_step = 0
        self.keep_prob = params.keep_prob
        # lr
        self.init_lr = params.lr
        self.lr_decay = params.lr_decay
        self.lr_decay_step = params.lr_decay_step
        self.warm_up_epoch=params.warm_up_step

        self.num_neg = params.num_neg
        self.num_layer=params.num_layer or 0
        # self.data_helper=data_helper
        self.eval_step_num = params.eval_step_num or 100
        self.eval_epoch_num = params.eval_epoch_num or 1
        self.log_step_num = params.log_step_num or 100
        self.saver = None


    def get_feed_dict(self, batch_datas, mode="train"):
        feed_dict = {}
        feed_dict[self.model.features["inputs"]["head"]] = batch_datas["inputs"]["head"]
        feed_dict[self.model.features["inputs"]["tail"]] = batch_datas["inputs"]["tail"]
        feed_dict[self.model.features["inputs"]["relation"]] = batch_datas["inputs"]["relation"]
        feed_dict[self.model.features["labels"]]=batch_datas["labels"]
        return feed_dict

    def train_step(self, sess, feed_dict):
        outputs = sess.run(self.model.train_outputs, feed_dict=feed_dict)
        return outputs
    def evaluate(self, sess, data_helper, evaluate_batch_num=None):
        data_gen = data_helper.train_batch_generator(batch_size=self.batch_size)
        count = 0.
        batch_count = 0
        losses = []
        accs=[]
        for batch_datas in data_gen:
            batch_count += 1
            if evaluate_batch_num is not None and batch_count >= evaluate_batch_num - 1:
                break
            feed_dict = self.get_feed_dict(batch_datas, mode="train")
            feed_dict[self.model.is_training] = False
            if "keep_prob" in self.model.features:
                feed_dict[self.model.features["keep_prob"]] = 1.0
            acc, loss = sess.run([self.model.acc, self.model.loss], feed_dict)
            losses.append(loss)
            accs.append(acc)
            batch_num = len(list(batch_datas.values())[0])
            count += batch_num
        acc = np.mean(accs)
        loss = sum(losses) / len(losses)
        return loss, acc
    def evaluate_model(self,sess, test_dataHelper, batch_size=64, mini_batch=8,test_num=100):
        ranks = []
        # predict
        gen = test_dataHelper.batch_generator(batch_size=batch_size,
                                              mini_batch=mini_batch)
        triples = test_dataHelper.triples
        for batch_datas in gen:
            if batch_datas is None:
                continue
            batch_pred_scores = self.predict_batch(sess,batch_datas)
            masks = batch_datas["masks"]
            inputs = batch_datas["inputs"]
            batch_tails = inputs["tail"]
            for tail, mask, preds in zip(batch_tails, masks, batch_pred_scores):
                preds=-preds
                tail_score = preds[tail]
                preds = preds + mask * (10000)
                preds[tail] = tail_score
                #
                score_tmp = preds[0]
                preds[0] = tail_score
                preds[tail] = score_tmp
                # 排序
                results = np.argsort(preds, axis=0)
                rank = np.where(results == 0)[0][0]
                ranks.append(rank)

            if test_num is not None and len(ranks)>=test_num*2:
                break
        print("test data:",len(ranks))
        mr,mrr,hit_1,hit_3,hit_10,hit_30=compute_metrics(ranks,is_print=True)
        self.log.info("Result: %s\t%s\t%s\t%s\t%s"%(mr,mrr,hit_1,hit_3,hit_10))
        return mrr,hit_10

    def predict(self, sess, data_helper):
        data_gen = data_helper.batch_generator(batch_size=self.batch_size)
        results = []
        start_time = time.time()
        for batch_datas in data_gen:
            output = self.predict_batch(sess, batch_datas)
            results.extend(output)
        end_time = time.time()
        print("time:%s" % (end_time - start_time))
        predicts = np.array(results)
        print("result shape:", predicts.shape)
        return predicts

    def predict_batch(self, sess, batch_datas):
        feed_dict = self.get_feed_dict(batch_datas, mode="predict")
        feed_dict[self.model.is_training] = False
        if "keep_prob" in self.model.features:
            feed_dict[self.model.features["keep_prob"]] = 1.0
        output = sess.run(self.model.scores, feed_dict)
        output = output[:, 0]
        return output

    def predict_embeddings(self, sess):
        entitiy_embeddings = sess.run(self.model.entity_embeddings)
        relation_embeddings = sess.run(self.model.relation_embeddings)
        return entitiy_embeddings, relation_embeddings

    def train(self, sess, data_helper, eval_data_helper, test_data_helper=None, iter_num=50, shuffle=True,
              use_early_stop=True, evaluate_batch_num=10):
        is_stop = False
        for epoch in range(iter_num):
            self.log.info("epoch: %s" % epoch)
            data_gen = data_helper.train_batch_generator(batch_size=self.batch_size, shuffle=shuffle,cur_epoch=epoch)
            total_loss = 0
            if epoch>=self.warm_up_epoch:
                self.lr = self.init_lr * (np.power(self.lr_decay, epoch // self.lr_decay_step))
                # self.lr = max(self.lr, 1e-8)
            else:
                self.lr=self.init_lr*((self.global_step+1)/(5*data_helper.data_num/self.batch_size))
            epoch_start_time=time.time()
            for batch_datas in data_gen:
                # start_time = time.time()
                feed_dict = self.get_feed_dict(batch_datas)
                feed_dict[self.model.lr] = self.lr
                feed_dict[self.model.is_training] = True
                if "keep_prob" in self.model.features:
                    feed_dict[self.model.features["keep_prob"]] = self.keep_prob

                # print(feed_dict)
                train_outputs = self.train_step(sess, feed_dict)
                loss = train_outputs["loss"]
                total_loss += loss
                acc = train_outputs["acc"]
                # end_time=time.time()
                self.global_step += 1
                # print("batch time: %s"%(end_time-start_time))
            if epoch % self.eval_epoch_num == 0:
                eval_loss, eval_acc = self.evaluate(sess, data_helper, evaluate_batch_num=evaluate_batch_num)
                eval_loss, eval_acc = self.evaluate(sess, eval_data_helper, evaluate_batch_num=evaluate_batch_num)
                if (epoch+1) %5==0:
                    eval_mr, hits10 = self.evaluate_model(sess, data_helper, test_num=50)
                    eval_mr, hits10 = self.evaluate_model(sess, eval_data_helper, test_num=50,mini_batch=8)
                    if test_data_helper is not None:
                        test_mr, hits10 = self.evaluate_model(sess, test_data_helper, test_num=None,mini_batch=8)
                        self.log.info("test mr:%s,test hits@10: %s" % (test_mr, hits10))
                self.log.info(
                    "epoch: %s, global step:%s, train_loss: %s,train_acc:%s" % (epoch, self.global_step, loss, acc))
                self.log.info("epoch: %s, global step:%s, valid loss: %s, valid acc: %s" % (
                epoch, self.global_step, eval_loss, eval_acc))
                print("epoch: %s, global step:%s, train_loss: %s,train_acc:%s" % (epoch, self.global_step, loss, acc))
                print("epoch: %s, global step:%s, valid loss: %s, valid acc: %s" % (
                epoch, self.global_step, eval_loss, eval_acc))

                self.save_weights(sess, global_step=epoch)
            epoch_end_time = time.time()
            print("lr: %s ,total loss :%s, time: %s"%(self.lr,total_loss, epoch_end_time-epoch_start_time))
            self.log.info("total loss: %s" % (total_loss))

    def save_weights(self, sess, global_step=None, saver=None):
        if saver is None:
            if self.saver is None:
                self.saver = tf.train.Saver(max_to_keep=100)
            saver = self.saver
        saver.save(sess, save_path=os.path.join(self.ckpt_path, "weights.ckpt"), global_step=global_step)

    def restore_last_session(self, sess):
        '''加载模型参数'''
        saver = tf.train.Saver()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore params from %s" % ckpt.model_checkpoint_path)
        else:
            print("fail to restore..., ckpt:%s" % ckpt)


class GATTrainer(BaseTrainer):

    def get_feed_dict(self,batch_datas,mode="train"):
        feed_dict={}
        feed_dict[self.model.features["inputs"]["head"]] = batch_datas["inputs"]["head"]
        feed_dict[self.model.features["inputs"]["tail"]] = batch_datas["inputs"]["tail"]
        feed_dict[self.model.features["inputs"]["relation"]] = batch_datas["inputs"]["relation"]
        #adjacency
        feed_dict[self.model.features["adjacency"]["head"]] = batch_datas["adjacency"]["head"]
        feed_dict[self.model.features["adjacency"]["tail"]] = batch_datas["adjacency"]["tail"]
        feed_dict[self.model.features["adjacency"]["relation"]] = batch_datas["adjacency"]["relation"]
        #nhop
        feed_dict[self.model.features["nhop"]["head"]] = batch_datas["nhop"]["head"]
        feed_dict[self.model.features["nhop"]["tail"]] = batch_datas["nhop"]["tail"]
        feed_dict[self.model.features["nhop"]["relation"]] = batch_datas["nhop"]["relation"]
        #labels
        feed_dict[self.model.features["labels"]] = batch_datas["labels"]
        feed_dict[self.model.features["label_masks"]] = batch_datas["label_masks"]
        feed_dict[self.model.features["label_masks_reverse"]] = batch_datas["label_masks_reverse"]
        return feed_dict
            
    def predict_embeddings(self,sess,data_helper):
        data_gen=data_helper.train_batch_generator(batch_size=self.batch_size)
        results=[]
        start_time=time.time()
        for batch_datas in data_gen:
            feed_dict = self.get_feed_dict(batch_datas)
            feed_dict[self.model.is_training] = False
            if "keep_prob" in self.model.features:
                feed_dict[self.model.features["keep_prob"]] = 1.0
            entity_embeddings,relation_embeddings,relation_embeddings_reverse = sess.run([self.model.final_entity_embeddings,
                                                              self.model.final_relation_embeddings,
                                                              self.model.final_relation_embeddings_reverse,], feed_dict)
            return entity_embeddings,relation_embeddings,relation_embeddings_reverse

    def evaluate_model(self,sess, test_dataHelper, batch_size=64, mini_batch=8,test_num=100):
        entity_embeddings, relation_embeddings ,relation_embeddings_reverse= self.predict_embeddings(sess, test_dataHelper)
        ranks = []
        # predict
        gen = test_dataHelper.batch_generator(batch_size=batch_size,
                                              mini_batch=mini_batch)
        triples = test_dataHelper.triples
        for batch_datas in gen:
            if batch_datas is None:
                continue
            batch_datas["entity_embedding"]=entity_embeddings
            batch_datas["relation_embedding"]=relation_embeddings
            batch_datas["relation_embedding_reverse"]=relation_embeddings_reverse
            batch_pred_scores,batch_pred_scores_reverse = self.predict_batch(sess,batch_datas)
            masks = batch_datas["masks"]
            masks_reverse = batch_datas["masks_reverse"]
            inputs = batch_datas["inputs"]
            batch_tails = inputs["tail"]
            batch_heads = inputs["head"]
            for tail, mask, preds in zip(batch_tails, masks, batch_pred_scores):
                preds=-preds
                tail_score = preds[tail]
                preds = preds + mask * (10000)
                preds[tail] = tail_score
                #
                score_tmp = preds[0]
                preds[0] = tail_score
                preds[tail] = score_tmp
                # 排序
                results = np.argsort(preds, axis=0)
                # results=result[::-1]
                # rank = np.where(results == tail)[0][0]
                rank = np.where(results == 0)[0][0]
                ranks.append(rank)
            #predict head
            for head, mask, preds in zip(batch_heads, masks_reverse, batch_pred_scores_reverse):
                preds=-preds
                head_score = preds[head]
                preds = preds + mask * (10000)
                preds[head] = head_score
                #
                score_tmp = preds[0]
                preds[0] = head_score
                preds[head] = score_tmp
                # 排序
                results = np.argsort(preds, axis=0)
                # results=result[::-1]
                # rank = np.where(results == tail)[0][0]
                rank = np.where(results == 0)[0][0]
                ranks.append(rank)

            if test_num is not None and len(ranks)>=test_num*2:
                break
        print("test data:",len(ranks))
        mr,mrr,hit_1,hit_3,hit_10,hit_30=compute_metrics(ranks,is_print=True)
        self.log.info("Result: %s\t%s\t%s\t%s\t%s"%(mr,mrr,hit_1,hit_3,hit_10))
        return mrr,hit_10

    def predict_batch(self, sess, batch_datas):
        feed_dict = self.get_feed_dict(batch_datas, mode="predict")
        feed_dict[self.model.is_training] = False
        feed_dict[self.model.features["entity_embedding"]]=batch_datas["entity_embedding"]
        feed_dict[self.model.features["relation_embedding"]]=batch_datas["relation_embedding"]
        feed_dict[self.model.features["relation_embedding_reverse"]]=batch_datas["relation_embedding_reverse"]
        if "keep_prob" in self.model.features:
            feed_dict[self.model.features["keep_prob"]] = 1.0
        output,output_reverse = sess.run([self.model.scores_predict,self.model.scores_predict_reverse], feed_dict)
        # output = output[:, :]
        return output,output_reverse