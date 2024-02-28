from __future__ import print_function, division
import os
import numpy as np
import torch
import logging

from utils import net_builder,count_parameters, get_logger
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from sklearn.metrics import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='SemiModel_wideresnet_HTRU_FAST_01')
    parser.add_argument('--version', type=str, default='ResNet18')
    parser.add_argument('--load_path', type=str, default='./saved_models')
    parser.add_argument('--use_train_model', type=str, default=True)
    parser.add_argument('--test', type=str, default=True)
    parser.add_argument('--num_labels', type=str, default='800_0124')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='ResNet18')# efficientnet WideResNet
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='../HTRU')
    parser.add_argument('-ds', '--dataset', type=str, default='HTRU')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--gpu', default=3, type=int,
                        help='GPU id to use.')
    args = parser.parse_args()

    # networks = ['WideResNet','WideResNetVar','ResNet18','ResNet34','ResNet50',
    #             'SEResNet18','SEResNet34','SEResNet50','efficientnet','MobileNet','MobileNetV2']
    # networks = ['WideResNet','WideResNetVar','ResNet18','ResNet34',
    #             'SEResNet18','SEResNet34','efficientnet',
    #             'MobileNet','MobileNetV2']
    # networks = ['WideResNet']
    networks = ['multimodel18']
    
    logger_level = "INFO"
    logger_save_path = os.path.join(args.save_dir, args.save_name,args.num_labels,'test')
    logger = get_logger(args.save_name, logger_save_path, logger_level)
    
    for net in networks[:]:
        parser.set_defaults(net=net)
        parser.set_defaults(version=net)
        args = parser.parse_args()
    # args = parser.parse_args()
        
        # save_path = os.path.join(args.save_dir, args.save_name,args.num_labels,args.version)
        save_path = os.path.join(args.save_dir, args.save_name,args.num_labels,args.version)
        

        # args.load_path = os.path.join(args.load_path, args.save_name,args.version,'model_best.pth')

        # args.load_path = os.path.join(args.load_path, args.save_name,args.num_labels,args.version)

        args.load_path = os.path.join(args.load_path, args.save_name,args.num_labels,args.version)
        # args.load_path = os.path.join(args.load_path, args.save_name,args.version) #fullysupervised-AdamW

        path_list = os.listdir(args.load_path)
        path_list.sort()
        for file in path_list:
            if os.path.splitext(file)[1] == '.pth':    
                checkpoint_path = os.path.join(args.load_path,file)
                logger.info('load model: '+ checkpoint_path)
                
                checkpoint = torch.load(checkpoint_path)
                load_model = checkpoint['model']

                

                
                _net_builder = net_builder(args.net, 
                                        args.net_from_name,
                                        {'depth': args.depth, 
                                            'widen_factor': args.widen_factor,
                                            'leaky_slope': args.leaky_slope,
                                            'dropRate': args.dropout,
                                            'use_embed': False})
                
                net = _net_builder(num_classes=args.num_classes)
                net.load_state_dict(load_model)
                logger.info(f'Number of test Params: {count_parameters(net)}')
                if torch.cuda.is_available():
                    net.cuda()
                net.eval()
                
                _eval_dset = SSL_Dataset(args=args,alg=args.save_name,name=args.dataset,
                                        num_classes=args.num_classes,data_dir=args.data_dir)
                eval_dset = _eval_dset.get_dset()
                
                eval_loader = get_data_loader(eval_dset,
                                            args.batch_size, 
                                            num_workers=0)
            
                acc = 0.0
                y_true = []
                y_pred = []
                y_logits = []
                with torch.no_grad():
                    for _,image, target in eval_loader:
                        image = image.type(torch.FloatTensor).cuda()
                        logit = net(image)
                        y_true.extend(target.cpu().tolist())           
                        y_pred.extend(torch.max(logit, dim=-1)[1].cpu().tolist())
                        y_logits.extend(torch.softmax(logit, dim=-1).cpu().tolist())
                        
                        # acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
                    pos_idx = np.where(np.array(y_true) == 0)[0]
                    sorted_idx = [i for i in np.argsort([y_logits[i][0] for i in range(len(y_logits))])[::-1]]
                    pos_ratio = 0.010
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
                    specificity = recall_score(y_true, y_pred, average='binary',pos_label=0)
                    F1 = f1_score(y_true, y_pred, average='binary',pos_label=0)
                    kappa = cohen_kappa_score(y_true, y_pred)
                    #print(y_true)
                    # AUC = roc_auc_score(y_true, y_pred, multi_class='ovo')
                    # fpr, tpr, thresholds = roc_curve(y_true, y_pred,pos_label=0)
                    #AUC = 'HTRU none'

                    cf_mat = confusion_matrix(y_true, y_pred, normalize=None)
                    logger.info('confusion matrix:\n' + np.array_str(cf_mat))
                    logger.info(f"前{pos_ratio*100:.2f}%中正样本占比: {pos_percentage:.5f}，最后一个正样本的百分位{pos_last_percentile:.5f}")

                    
                
                logger.info('Test Accuracy: {:.5f}, recall: {:.5f}, specificity: {:.5f}, precision: {:.5f}, F1: {:.5f}, kappa:{:.5f}'.format(
                    top1,recall,specificity,precision,F1,kappa
                ))
                # logging.shutdown()
