from __future__ import print_function, absolute_import
import time
import torch
import torch.nn as nn
import numpy as np
from utils import AverageMeter
from torch.nn import functional as F
from loss import TripletLoss_WRT,NEL
import os

class Trainer(object):
    def __init__(self, model):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = nn.CrossEntropyLoss().cuda()
        self.criterion_triple = TripletLoss_WRT().cuda()
        self.T = 2

    def train(self, epoch, data_loader_train, data_loader_replay, optimizer, training_phase,
              add_num=0, old_model=None, replay=False,writer = None):
        self.model.train()
        if old_model is not None:
            old_model.train()
            old_model.freeze_all()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        id_loss = AverageMeter()
        tp_loss = AverageMeter()
        kl_loss = AverageMeter()
        id_loss_specific = AverageMeter()
        goal_loss = AverageMeter()
        losses_total = AverageMeter()
        losses_pull = AverageMeter()
        losses_ce_general = AverageMeter()
        rehersal_loss = AverageMeter()
        tp_loss_r = AverageMeter()

        end = time.time()

        for batch_idx, (input1, input2, label1, label2) in enumerate(data_loader_train):

            data_time.update(time.time() - end)

            labels = torch.cat((label1, label2), dim=0)
            labels += add_num

            # delete the seq_len dim
            # [B,seq_len,C,H,W] --> [B,C,H,W]
            if len(input1.size()) == 5:
                input1 = input1.squeeze(1)
                input2 = input2.squeeze(1)

            input1 = input1.cuda()
            input2 = input2.cuda()

            labels = labels.cuda()
            label1,label2 = labels.chunk(2,0)

            out = self.model(input1, input2,training_phase = training_phase)

            loss_ce, loss_tp, loss_kd = self._forward(out["feat"], out["logits"], labels-add_num)

            loss_goal = loss_ce + loss_tp + loss_kd

            lamba = sum(self.model.class_per_task[:-1]) / sum(self.model.class_per_task)

            loss_total = (1-lamba) * loss_goal

            if self.model.head_div is not None:
                if training_phase == 1:
                    loss_ce_specific = self.criterion_ce(out["div_logits"], labels)
                else:
                    loss_ce_specific = self.criterion_ce(out["div_logits"], labels-add_num+1)

            if self.model.KeyToken:
                similarity = 1 - torch.matmul(out["query"], out["current_key"].t())  # B,1
                pull_loss = torch.sum(similarity) / out["query"].shape[0]  # scaler

            if self.model.GeneralToken:
                loss_ce_general = self.criterion_ce(out["general_logits"], labels)

            id_loss.update(loss_ce, 2 * input1.size()[0])
            tp_loss.update(loss_tp)
            kl_loss.update(loss_kd)
            goal_loss.update(loss_goal)

            if replay is True:
                input1_r, input2_r, labels1_r, labels2_r,domain_r = next(iter(data_loader_replay))
                domain = int(domain_r[0])
                input1_r = input1_r.cuda()
                input2_r = input2_r.cuda()

                labels_r = torch.cat((labels1_r, labels2_r), dim=0)
                labels_r = labels_r.cuda()
                labels1_r,labels2_r = labels_r.chunk(2,0)

                out_r = self.model(input1_r, input2_r,training_phase = training_phase)

                if self.model.GeneralToken:
                    loss_ce_general += self.criterion_ce(out_r["general_logits"], labels_r)

                if self.model.KeyToken:
                    old_keys = out_r["old_keys"][domain-1] #1,embed_dim
                    similarity = 1 - torch.matmul(out_r["query"],old_keys.t()) # B,1
                    pull_loss += torch.sum(similarity) / out_r["query"].shape[0]

                    out_old = old_model(input1_r,input2_r,training_phase = training_phase)

                    loss_kd = self.loss_kd_js(out_old["old_logits"][domain-1],out_r["old_logits"][domain-1])

                    loss_tp_r = self.criterion_triple(out_r["old_feats"][domain-1],labels_r)[0]

                    loss_rehersal = (1-lamba) * loss_tp_r + lamba * loss_kd

                else:
                    loss_tr_r, _ = self.criterion_triple(out_r["feat"], labels_r)
                    out_old = old_model(input1_r, input2_r, training_phase=training_phase)

                    loss_kd = self.loss_kd_js(out_old["logits"][:add_num], out_r["logits"][:add_num])
                    tp_loss_r.update(loss_tr_r)

                    loss_rehersal = (1 - lamba) * loss_tr_r + lamba * loss_kd

                if self.model.head_div is not None:
                    loss_ce_specific += self.criterion_ce(out_r["div_logits"],
                                                  torch.zeros_like(labels_r, dtype=torch.long))

                rehersal_loss.update(loss_rehersal)
                loss_total += loss_rehersal
                del out,out_r,out_old

            if self.model.GeneralToken:
                loss_total += 0.1 * loss_ce_general
                losses_ce_general.update(loss_ce_general)

            if self.model.KeyToken:
                loss_total += 0.1 * pull_loss
                losses_pull.update(pull_loss)

            if self.model.head_div is not None:
                loss_total += 0.1 * loss_ce_specific
                id_loss_specific.update(loss_ce_specific, 2 * input1.size()[0])

            losses_total.update(loss_total)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx + 1) == len(data_loader_train) or (batch_idx + 1) % (len(data_loader_train) // 4) == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_goal {:.3f} ({:.3f})\t'
                      'Loss_id_div {:.3f}({:.3f})\t'
                      'Loss_id_general {:.3f}({:.3f})\t'
                      'Loss_pull {:.3f}({:.3f})\t'
                      'Loss_total {:.3f} ({:.3f})\t'
                      'Loss_rehearsal {:.3f} ({:.3f})\t'
                      .format(epoch, batch_idx + 1, len(data_loader_train),
                              batch_time.val, batch_time.avg,
                              goal_loss.val, goal_loss.avg,
                              id_loss_specific.val, id_loss_specific.avg,
                              losses_ce_general.val, losses_ce_general.avg,
                              losses_pull.val, losses_pull.avg,
                              losses_total.val, losses_total.avg,
                              rehersal_loss.val, rehersal_loss.avg))
            if writer is not None:
                writer.add_scalar('total_loss', losses_total.val,batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('id_loss', id_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('kl_loss', kl_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('id_loss_div', id_loss_specific.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('tri_loss', tp_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('general_loss', losses_ce_general.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('pull_loss', losses_pull.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('rehearsal_loss', rehersal_loss.val, batch_idx + epoch * len(data_loader_train))


    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr, _ = self.criterion_triple(s_features, targets)
        n = s_outputs.size()[0] // 2
        feat1 = s_outputs.narrow(0, 0, n)
        feat2 = s_outputs.narrow(0, n, n)
        loss_kd = self.loss_kd_js(feat1, feat2)

        return loss_ce, loss_tr, loss_kd

    def loss_kd_js(self, old_logits, new_logits):
        old_logits = old_logits.detach()
        p_s = F.log_softmax((new_logits + old_logits) / (2 * self.T), dim=1)
        p_t = F.softmax(old_logits / self.T, dim=1)
        p_t2 = F.softmax(new_logits / self.T, dim=1)
        loss = 0.5 * F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2) + 0.5 * F.kl_div(p_s, p_t2,reduction='batchmean') * (self.T ** 2)

        return loss

