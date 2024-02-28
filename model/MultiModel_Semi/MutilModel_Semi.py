import numpy as np
import torch
import torch.nn.functional as F
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import EMA, Bn_Controller
from collections import Counter
from train_utils import ce_loss, wd_loss
from model.MultiModel_Semi.MutilModel_Semi_util import AdaptiveMultiModalConsistencyLoss,MultiModalSupervisedLoss,consistency_loss,Supervised_label

from copy import deepcopy
from sklearn.metrics import *
import math
import pickle # add
import logging
import time
from pytorch_msssim import SSIM,ssim

# logging.basicConfig(filename='weights_log.txt', level=logging.INFO) 

class MultiModel_Semi:
    def __init__(self, net_builder, num_classes, lambda_u,
                 num_eval_iter=1000, tb_log=None, ema_m=0.999, logger=None):
        """
        class MultiModel_Semi contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            lambda_u: ratio of unsupervised loss to supervised loss
            it: initial iteration count
            num_eval_iter: frequency of evaluation.
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(MultiModel_Semi, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes

        self.lambda_max = 0.80  # 设置 lambda_max
        self.mu_t = 0.90  # 设置 mu_t
        self.mu_t_min = 0.75  # 设置 mu_t 的最小值
        self.mu_t_max = 0.99  # 设置 mu_t 的最大值
        self.mu_t_rate = 0.01  # 设置  mu_t 的学习率
        self.performance_window = []  # 初始化性能滑动窗口
        self.window_size = 10  # 设置滑动窗口大小
        self.sigma_t = 0.1  # 设置 sigma_t

        self.ema_m = ema_m
        self.ema_p = 0.999
        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.alpha = 0.1

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.un_loss = AdaptiveMultiModalConsistencyLoss()  # 无标签损失
        # self.loss = MultiModalSupervisedLoss()              # 有标签损失

        self.bn_controller = Bn_Controller()

        flop = self.calculate_flops()
        # self.print_fn('flop=',flop)
    


    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')
        

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    #     # 计算预测置信度
    # def confidence(self,probs):
    #     return probs.max(dim=1)[0]

    # 计算权重
    def compute_weights(self, probs, x_ulb_w, mu_t, sigma_t, lambda_max, sigma_s):
        max_probs = probs.max(dim=1)[0]
        weights = torch.where(
            max_probs < mu_t,
            lambda_max * torch.exp(-((max_probs - mu_t)**2) / (2 * sigma_t**2)),
            torch.tensor([lambda_max]).to(probs.device)
        )
        min_max_probs = max_probs.min()
        max_max_probs = max_probs.max()
        with open('max_probs_stats.txt', 'a') as f:
            f.write(f'max_probs Min: {min_max_probs}  ')
            f.write(f'max_probs Max: {max_max_probs}\n')
        return weights
    
    # 加入了方差
    # def compute_weights(self,probs, variance, mu_t, sigma_t, lambda_max, alpha):
    #     max_probs = probs.max(dim=1)[0]
    #     weights = torch.where(
    #         max_probs < mu_t,
    #         lambda_max * torch.exp(-((max_probs - mu_t)**2) / (2 * (sigma_t**2 + alpha * variance))),
    #         torch.tensor([lambda_max]).to(probs.device)
    #     )
    #     return weights
    
    # # 加入了样本相似度
    # def compute_weights(self, probs, x_ulb_w, mu_t, sigma_t, lambda_max, sigma_s):
    #     max_probs = probs.max(dim=1)[0]
    #     # 计算排名
    #     ranks = max_probs.argsort().argsort()
    #     # 归一化排名
    #     normalized_ranks = ranks.float() / (ranks.size(0) - 1)
    #     weights = torch.where(
    #         normalized_ranks < mu_t,
    #         lambda_max * torch.exp(-((normalized_ranks - mu_t)**2) / (2 * sigma_t**2 ))
    #         * self.compute_similarity(x_ulb_w),
    #         torch.tensor([lambda_max]).to(probs.device)
    #     )
    #     # logging.info(f'[compute_similarity {self.compute_similarity(x_ulb_w)}] ,\n'
    #     #     )
    #     return weights
    
    # 去掉了样本相似度
    # def compute_weights(self, probs, x_ulb_w, mu_t, sigma_t, lambda_max, sigma_s):
    #     max_probs = probs.max(dim=1)[0]
        
    #     # 计算排名
    #     ranks = max_probs.argsort().argsort()
    #     # # 归一化排名
    #     normalized_ranks = ranks.float() / (ranks.size(0) - 1)
    #     # ranks_exp = ranks.float().pow(2)  # 指数增加差异
    #     # normalized_ranks_exp = ranks_exp / ranks_exp.max()
    #     weights = torch.where(
    #         normalized_ranks < mu_t,
    #         lambda_max * torch.exp(-((normalized_ranks - mu_t)**2) / (2 * sigma_t**2 )),
    #         torch.tensor([lambda_max]).to(probs.device)
    #     )
    #     # logging.info(f'[compute_similarity {self.compute_similarity(x_ulb_w)}] ,\n'
    #     #     )
    #     return weights

    # def compute_weights(self, probs, x_ulb_w, mu_t, sigma_t, lambda_max, sigma_s):
    #     max_probs = probs.max(dim=1)[0]
        
    #     # 计算排名
    #     ranks = max_probs.argsort().argsort()
    #     # # 归一化排名
    #     # 确定原始max_probs的最大值和最小值用于缩放排名
    #     min_max_probs = max_probs.min()
    #     max_max_probs = max_probs.max()
        
    #     # 将排名缩放到max_probs的最大值和最小值之间
    #     # scaled_ranks = min_max_probs + (max_max_probs - min_max_probs) * torch.log(ranks.float() + 1) / torch.log(torch.tensor(ranks.size(0)))
    #     scaled_ranks = min_max_probs + (max_max_probs - min_max_probs) * torch.sqrt(ranks.float()) / torch.sqrt(torch.tensor(ranks.size(0) - 1))
    #     weights = torch.where(
    #         scaled_ranks < mu_t,
    #         lambda_max * torch.exp(-((scaled_ranks - mu_t)**2) / (2 * sigma_t**2 )),
    #         torch.tensor([lambda_max]).to(probs.device)
    #     )

    #     # 将scaled_ranks的最大值和最小值保存到txt文件中
    #     scaled_ranks_min = scaled_ranks.min().item()
    #     scaled_ranks_max = scaled_ranks.max().item()
        
    #     with open('log_scaled_ranks_stats.txt', 'a') as f:
    #         f.write(f'max_probs Min: {min_max_probs}  ')
    #         f.write(f'max_probs Max: {max_max_probs}  ')
    #         f.write(f'Scaled Ranks Min: {scaled_ranks_min}  ')
    #         f.write(f'Scaled Ranks Max: {scaled_ranks_max}\n')
    #         # logging.info(f'[compute_similarity {self.compute_similarity(x_ulb_w)}] ,\n'
    #         #     )
    #     return weights
    
    # def compute_similarity(self, x_i):
    #     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #     x_i = x_i.to(device)
    #     # 获取样本的类别
    #     class_labels = self.model(x_i).argmax(dim=1)
    #     similarities = torch.zeros(x_i.size(0)).to(device)
    #     for i in range(x_i.size(0)):     
    #         class_label = class_labels[i]
    #         # 获取同一类别的所有样本
    #         X = self.samples_by_class[class_label.item()].to(device)
    #         similarity_values = torch.zeros(X.size(0)).to(device)
    #         for j in range(X.size(0)):
    #             similarity_values[j] = ssim(X[j].unsqueeze(0), x_i[i].unsqueeze(0), win_size=11, win_sigma=1.5, K=(0.01,0.03), size_average=True)
    #         similarities[i] = similarity_values.mean()
    #     return similarities

    # # Cosine Similarity
    # def compute_similarity(self, x_i):
    #     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #     x_i = x_i.to(device)
    #     # 获取样本的类别
    #     class_labels = self.model(x_i).argmax(dim=1)
    #     similarities = torch.zeros(x_i.size(0)).to(device)
    #     for i in range(x_i.size(0)):     
    #         class_label = class_labels[i]
    #         # 获取同一类别的所有样本
    #         X = self.samples_by_class[class_label.item()].to(device)
    #         similarity_values = torch.zeros(X.size(0)).to(device)
    #         for j in range(X.size(0)):
    #             # 计算 Cosine Similarity
    #             similarity_values[j] = F.cosine_similarity(X[j].unsqueeze(0).view(1, -1), x_i[i].unsqueeze(0).view(1, -1))
    #         similarities[i] = similarity_values.mean()
    #     return similarities

    ## 直方图比较
    # def compute_similarity(self, x_i):
    #     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #     x_i = x_i.to(device)
    #     # 获取样本的类别
    #     class_labels = self.model(x_i).argmax(dim=1)
    #     similarities = torch.zeros(x_i.size(0)).to(device)
    #     for i in range(x_i.size(0)):     
    #         class_label = class_labels[i]
    #         # 获取同一类别的所有样本
    #         X = self.samples_by_class[class_label.item()].to(device)
    #         similarity_values = torch.zeros(X.size(0)).to(device)
    #         for j in range(X.size(0)):
    #             # 计算直方图
    #             hist_x_i = torch.histc(x_i[i], bins=256, min=0, max=255)
    #             hist_X_j = torch.histc(X[j], bins=256, min=0, max=255)
    #             # 计算直方图的相似度
    #             similarity_values[j] = torch.sum((hist_x_i - hist_X_j) ** 2) ** 0.5
    #         similarities[i] = similarity_values.mean()
    #     return similarities
    
    # # 使用 dropout 的方法来估计预测的方差
    # def estimate_variance(self, model, x_ulb_w, n_samples=10):
    #     model.train()
    #     predictions = [model(x_ulb_w) for _ in range(n_samples)]
    #     predictions = torch.stack(predictions)
    #     variance = torch.var(predictions, dim=0).mean(dim=1)  # 在类别维度上取平均
    #     model.eval()
    #     return variance

    # def update_prob_t(self, lb_probs, ulb_probs):
    #     ulb_prob_t = ulb_probs.mean(0)
    #     self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

    #     lb_prob_t = lb_probs.mean(0)
    #     self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

    #     max_probs, max_idx = ulb_probs.max(dim=-1)
    #     prob_max_mu_t = torch.mean(max_probs)
    #     prob_max_var_t = torch.var(max_probs, unbiased=True)
    #     self.mu_t = self.ema_p * self.mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
    #     self.sigma_t = self.ema_p * self.sigma_t + (1 - self.ema_p) * prob_max_var_t.item()
    
    # def compute_weights(self,probs, mu_t, sigma_t, lambda_max):
    #     max_probs = probs.max(dim=1)[0]
    #     # 计算排名
    #     ranks = max_probs.argsort().argsort()
    #     # 归一化排名
    #     normalized_ranks = ranks.float() / (ranks.size(0) - 1)
    #     weights = torch.where(
    #         normalized_ranks < mu_t,
    #         lambda_max * torch.exp(-((normalized_ranks - mu_t)**2) / (2 * sigma_t**2)),
    #         torch.tensor([lambda_max]).to(probs.device)
    #     )
    #     logging.info(f'[mu_t {(mu_t)}] ,\n'
    #                  f'[max_probs {(max_probs)}] ,\n'
    #                  f'[mu_t {(mu_t)}] ,\n'
    #         f'[weights {(weights)}] ,\n'
    #     f'[normalized_ranks {(normalized_ranks)}] ,\n'
    #          f'exp {torch.exp(-((normalized_ranks - mu_t)**2) / (2 * sigma_t**2 ))}')
    #     return weights

    def collect_samples_by_class(self):
        # print('***************samples*******************')
        # Initialize the dictionary of samples by class
        self.samples_by_class = {}
        # Iterate over the entire labeled training set
        n = 50
        for i, ( _, x_lb, y_lb ) in enumerate(self.loader_dict['train_lb']):
            # Flatten the batch dimension
            x_lb = x_lb
            y_lb = y_lb.view(-1)
            # print(y_lb.shape)
            # Add each sample to its class's list
            for x, y in zip(x_lb, y_lb):
                class_label = y.item()
                if class_label not in self.samples_by_class:
                    self.samples_by_class[class_label] = []
                self.samples_by_class[class_label].append(x)
            if i >= n - 1:
                break
        # Convert the lists to tensors
        for class_label in self.samples_by_class:
            self.samples_by_class[class_label] = torch.stack(self.samples_by_class[class_label])
    
    # def compute_sample_difficulty_unlabeled(self, logits):
    #     # 这里我们假定难度与模型对最可能类别的预测置信度成反比
    #     probs = torch.softmax(logits, dim=1)
    #     max_probs, _ = probs.max(dim=1)
    #     difficulty = 1 - max_probs
    #     return difficulty
    
    # def curriculum_learning(self, logits, difficulty_threshold=0.6):
    #     # 计算无标签样本的难度
    #     difficulty = self.compute_sample_difficulty_unlabeled(logits)

    #     # 根据难度进行排序
    #     sorted_indices = torch.argsort(difficulty)

    #     # 根据当前的难度阈值选择样本
    #     selected_indices = sorted_indices[difficulty[sorted_indices] < difficulty_threshold]

    #     return selected_indices
    
    def calculate_flops(self):

        model = self.model
        input_size = (1, 3, 32, 32)
        # 定义模型输入尺寸
        input_tensor = torch.randn(input_size)

        # 输出模型摘要，并获取 FLOPs
        model_summary = summary(
            model,
            input_data=input_tensor,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        )
        flops = model_summary.total_mult_adds
        return flops

    def train(self, args):

        ngpus_per_node = torch.cuda.device_count()

        # EMA init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it,best_pos_percentage,best_it_pos,best_eval_F1,best_eval_kappa = 0.0, 0.0, 0.0,0.0,0.0,0.0
        best_it_kappa = 0.0
        best_it_acc = 0

        starting_threshold = 0.2
        ending_threshold = 1.0
        difficulty_threshold = 0.9
        decay_rate = 0.9
        

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        # selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        # selected_label = selected_label.cuda(args.gpu)
        # classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

        # p_model = (torch.ones(args.num_classes) / args.num_classes).cuda()
        # label_hist = (torch.ones(args.num_classes) / args.num_classes).cuda() 
        # time_p = p_model.mean()

        
        # times = 0
        # self.sample_weights_history = {} # add
        # num_ulb_samples = len(self.loader_dict['train_ulb'].dataset) # add
        # for idx in range(num_ulb_samples): # add
        #     self.sample_weights_history[idx] = [] # add

        self.collect_samples_by_class()
        weight_sum = 0
        for (i, x_lb, y_lb), (x_ulb_idx, x_ulb_w) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            # print('******************start******************')  
            # print(self.samples_by_class)
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            # unsup_warmup = np.clip(self.it / (args.unsup_warmup_pos * args.num_train_iter),
            #                        a_min=0.0, a_max=1.0)
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            x_lb, x_ulb_w = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu)
            self.x_lb = x_lb.cuda(args.gpu)
            self.x_ulb_w = x_ulb_w.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            num_lb = x_lb.shape[0]
            input = torch.cat((x_lb,x_ulb_w))

            

            # self.samples_by_class = {}
            # for x, y in zip(x_lb, y_lb):
            #     class_label = y.item()
            #     if class_label not in self.samples_by_class:
            #         self.samples_by_class[class_label] = []
            #     self.samples_by_class[class_label].append(x)
            # for class_label in self.samples_by_class:
            #     self.samples_by_class[class_label] = torch.stack(self.samples_by_class[class_label])

            

            
            # if args.use_flex:
            #     pseudo_counter = Counter(selected_label.tolist())
            #     if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
            #         for i in range(args.num_classes):
            #             classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            # inference and calculate sup/unsup losses
            # with amp_cm():

            # logits_x_lb,logits_x1_lb,logits_x2_lb = self.model(x_lb, return_modal_outputs=True)
            logits = self.model(input)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w = logits[num_lb:]
            # _,logits_x_lb = Supervised_label(logits_x_lb,logits_x1_lb,logits_x2_lb)
            self.bn_controller.freeze_bn(self.model)
            self.bn_controller.unfreeze_bn(self.model)

            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # self.update_prob_t(probs_x_lb, probs_x_ulb_w)

            # 有标签数据损失
            # sup_loss = self.loss(logits_x1_lb,logits_x2_lb,logits_x3_lb, y_lb)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            
            # # 计算预测结果的方差
            # variance = self.estimate_variance(self.model, x_ulb_w)

            # 需要确定无标签数据的损失怎么计算
            # 伪标签数据损失
            pseudo_labels = logits_x_ulb_w.argmax(dim=1)
            probs = torch.softmax(logits_x_ulb_w, dim=1)
            
            # print('******************weights start******************')
            # weights = self.compute_weights(probs, self.mu_t, self.sigma_t, self.lambda_max)
            weights = self.compute_weights(probs, self.x_ulb_w, self.mu_t, self.sigma_t, self.lambda_max, sigma_s=0.5)
            # print('******************weights end******************')

            weights_gt_08 = (weights >= self.lambda_max).sum().item()
            prob_weight = weights_gt_08/len(weights)
            # weight_sum = weight_sum + weights_gt_08
            if self.it % 100 == 0 or self.it<100:

                with open('weights_HTRU_FAST_0221.txt', 'a') as file:
                    file.write(f'it: {self.it},weights: {weights_gt_08},prob_weights: {prob_weight}, mu_t: {self.mu_t}\n') 
            

            # 以日志形式记录
            torch.set_printoptions(precision=20)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            # logging.info(f'[{current_time}]: {weights},  \n    {self.mu_t}')
            # num_lable = int(x_ulb_w.shape[0])
            # for idx in np.arange(num_lable*times,num_lable*(times+1)): # add
            #     weights_idx = idx%num_lable
            #     self.sample_weights_history[idx].append(weights[weights_idx].item()) # add

            # mask = weights >= 0.80
            # selected_logits = logits_x_ulb_w[mask]
            # selected_labels = pseudo_labels[mask]

            # print(ce_loss(logits_x_ulb_w, pseudo_labels) )
            unsup_loss = (ce_loss(logits_x_ulb_w, pseudo_labels) * weights).mean()
          

            # times = times+1
            # selected_indices = self.curriculum_learning(logits_x_ulb_w, difficulty_threshold)
            # pseudo_labels = logits_x_ulb_w[selected_indices].argmax(dim=1)
            # probs = torch.softmax(logits_x_ulb_w[selected_indices], dim=1)
            # weights = self.compute_weights(probs, self.mu_t, self.sigma_t, self.lambda_max)
            # unsup_loss = (ce_loss(logits_x_ulb_w[selected_indices], pseudo_labels) * weights).mean()

            

            total_loss = sup_loss + unsup_loss
            # print(sup_loss,'\n',unsup_loss)

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # save model for each 10K steps and best model for each 1K steps
            if self.it % self.num_eval_iter == 0 and self.it>1000:
                save_path = os.path.join(args.save_dir, args.save_name,args.num_labels,args.version)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:

                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name,args.num_labels,args.version)

                current_eval_F1 = tb_dict['eval/recall']  #tb_dict['eval/F1']
                self.performance_window.append(current_eval_F1)
                if len(self.performance_window) > self.window_size:
                    self.performance_window.pop(0)  # 保持窗口大小
                smoothed_F1 = sum(self.performance_window) / len(self.performance_window)  # 计算滑动窗口平均
                if smoothed_F1 >= current_eval_F1:
                    # 如果平滑后的 F1 分数比最佳的 F1 分数高，那么提高 mu_t
                    self.mu_t = min(self.mu_t_max, self.mu_t + self.mu_t_rate)
                    best_eval_F1 = smoothed_F1
                else:
                    # 如果平滑后的 F1 分数没有提高，那么降低 mu_t
                    self.mu_t = max(self.mu_t_min, self.mu_t - self.mu_t_rate)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it_acc = self.it
                if tb_dict['eval/pos_percentage'] > best_pos_percentage:
                    best_pos_percentage = tb_dict['eval/pos_percentage']
                    best_it_pos = self.it
                if tb_dict['eval/F1'] > best_eval_F1:
                    best_eval_F1 = tb_dict['eval/F1']
                    best_it = self.it
                #     difficulty_threshold = min(ending_threshold, difficulty_threshold + 0.05)
                # else:
                #     difficulty_threshold = max(starting_threshold, difficulty_threshold * decay_rate)
                # if tb_dict['eval/kappa'] >= best_eval_kappa:
                #     best_eval_kappa = tb_dict['eval/kappa']
                #     best_it_kappa = self.it
                


                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict},\n BEST_EVAL_acc: {best_eval_acc}, at {best_it_acc} iters, 正样本占比：{best_pos_percentage}, at {best_it_pos} iters, BEST_EVAL_F1: {best_eval_F1}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it or tb_dict['eval/F1']>0.94:
                        self.save_model('latest_model_{}.pth'.format(self.it), save_path)
                        if self.it == best_it:
                            self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            
                  
                

            
            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000
        
        # save_path = os.path.join('/aidata/Ly61/number3/python/', 'sample_weights_history.pkl') 
        # with open(save_path, 'wb') as f: # add 
        #     pickle.dump(self.sample_weights_history, f) # add
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            # logits,logits_x1_lb_eva,logits_x2_lb_eva,logits_x3_lb_eva = self.model(x, return_modal_outputs=True)
            # pre,logits = Supervised_label(logits_x1_lb_eva,logits_x2_lb_eva,logits_x3_lb_eva)
            # loss = F.cross_entropy(logits, y.to(torch.float32), reduction='mean')
            # y_true.extend(y.cpu().tolist())
            # y_pred.extend(pre.cpu().tolist())
            # y_logits.extend(logits.cpu().tolist())
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
            # if len(y_true) > 540:
            #     break
        pos_idx = np.where(np.array(y_true) == 0)[0]
        sorted_idx = [i for i in np.argsort([y_logits[i][0] for i in range(len(y_logits))])[::-1]]
        pos_ratio = 0.25 # 0.246
        sample_count = int(len(y_logits) * pos_ratio)
        top_sample = sorted_idx[:sample_count]
        pos_percentage = len(set(top_sample).intersection(set(pos_idx))) / len(top_sample)

        ## 排名最后的正样本的百分位
        last_pos_rank = len(y_logits) - 1  
        for idx in sorted_idx[::-1]:
            if idx in pos_idx:
                last_pos_rank = sorted_idx.index(idx)
                break
        pos_last_percentile = last_pos_rank / len(sorted_idx)

        top1 = accuracy_score(y_true, y_pred)
        #top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        #top5 = 'HTRU none'
        precision = precision_score(y_true, y_pred, average='binary',pos_label=0)
        recall = recall_score(y_true, y_pred, average='binary',pos_label=0)
        _F1 = f1_score(y_true, y_pred, average='binary',pos_label=0)
        # kappa = cohen_kappa_score(y_true, y_pred)
        #print(y_true)
        
        # AUC = roc_auc_score(y_true, y_pred, multi_class='ovo')
        # AUC = 'none'

        cf_mat = confusion_matrix(y_true, y_pred, normalize=None)
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.print_fn(f"前{pos_ratio*100:.2f}%中正样本占比: {pos_percentage:.5f}，最后一个正样本的百分位{pos_last_percentile:.5f}")
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': _F1,'eval/pos_percentage':pos_percentage}
        
    def save_model(self, save_name, save_path):
        # if self.it < 1000000:
        #     return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    @torch.no_grad()
    def cal_time_p_and_p_model(self,logits_x_ulb_w, time_p, p_model, label_hist):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        if label_hist is None:
            label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist / label_hist.sum()
        else:
            hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p,p_model,label_hist
    
    
    # Abandoned in Pseudo Label
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    


if __name__ == "__main__":
    pass
