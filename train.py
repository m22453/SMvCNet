
import math
import sys
from typing import Iterable
import os

import numpy as np
import torch

from sklearn.cluster import KMeans
import utils
from utils import adjust_learning_config, SmoothedValue, MetricLogger



def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    # model training
    model.train(True)
    optimizer.zero_grad()

    for data_iter_step, (ids, datas, mask, _) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            datas[i] = datas[i].to(device, non_blocking=True)

        with torch.autocast('cuda', enabled=False):
            loss = model(datas, mmt, epoch < args.start_rectify_epoch)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # sys.exit(1)
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    if args.print_this_epoch:
        print("Averaged stats:", metric_logger)
        eval_result = evaluate(model, data_loader_test, device, epoch, args)
    else:
        eval_result = None
    return eval_result


def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device, epoch: int,
             args=None):
    model.eval()
    extracter = model.extract_feature
    path = os.path.join(args.output_dir, 'visualize')
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        for indexs, datas, mask, labels in data_loader_test:
            
            for i in range(args.n_views):
                datas[i] = datas[i].to(device, non_blocking=True)
            # np.save(path+'/embeddings_ours_data_0.npy', np.hstack((datas[0].cpu(), datas[1].cpu())))
            

            labels = labels.to(device, non_blocking=True)
            features = extracter(datas, mask)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]
            

            labels_all[indexs] = labels 
 
        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()
        kmeans_label = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(features_cat)
        # if epoch == args.start_rectify_epoch - 1 or epoch == args.epochs -1 or epoch == 0:
        #     try:
                
        #         if args.mvrw_weights[0]:
        #             path_data = path + '/embeddings_ours_{}.npy'.format(epoch)
        #             path_label = path + '/embeddings_ours_label.npy'
        #             for i in range(args.n_views):
        #                 np.save(path+'/embeddings_ours_view{}_{}'.format(i, epoch), features_all[i].cpu())
        #         else:
        #             path_data = path + '/embeddings_divide_{}.npy'.format(epoch)
        #             path_label = path + '/embeddings_divide_label.npy'
        #         np.save(path_data, features_cat)
        #         np.save(path_label, np.asarray(labels_all.cpu()))
                
        #     except Exception as e:
        #         print("break for :"+str(e))
        
    nmi, ari, f, acc, pur = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_label)
    result = {'nmi': nmi, 'ari': ari, 'f': f, 'acc': acc, 'purity': pur}
    return result
