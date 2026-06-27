from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, recall_score, precision_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from utils import *
import torch
import tqdm
from loguru import logger
import joblib
import os

class ClassMetrics:
    def __init__(self):
        self.acc_list = []
        self.f1_list = []
        self.mcc_list = []
        self.recall_list = []
        self.precision_list = []
        self.rocauc_list = []
        self.prauc_list = []
    def add_metric(self, acc, f1, mcc, recall, precision, rocauc, prauc):
        self.acc_list.append(acc)
        self.f1_list.append(f1)
        self.mcc_list.append(mcc)
        self.recall_list.append(recall)
        self.precision_list.append(precision)
        self.rocauc_list.append(rocauc)
        self.prauc_list.append(prauc)
    def get_ave_metrics(self):
        return np.mean(self.acc_list), np.mean(self.f1_list), np.mean(self.mcc_list), np.mean(self.recall_list), np.mean(self.precision_list), np.mean(self.rocauc_list), np.mean(self.prauc_list)
    def get_cur_metrics(self):
        return self.acc_list[-1], self.f1_list[-1], self.mcc_list[-1], self.recall_list[-1], self.precision_list[-1], self.rocauc_list[-1], self.prauc_list[-1]
    
class RegressMetrics:
    def __init__(self):
        self.mae_list = []
        self.mse_list = []
        self.rmse_list = []
        self.r2_list = []
    def add_metric(self, mae, mse, rmse, r2):
        self.mae_list.append(mae)
        self.mse_list.append(mse)
        self.rmse_list.append(rmse)
        self.r2_list.append(r2)
    def get_ave_metrics(self):
        return np.mean(self.mae_list), np.mean(self.mse_list), np.mean(self.rmse_list), np.mean(self.r2_list)
    def get_cur_metrics(self):
        return self.mae_list[-1], self.mse_list[-1], self.rmse_list[-1], self.r2_list[-1]
'''
计算prauc，可以用`average_precision_score`，也可以用`auc`，二者的计算方式不同，有时会得到不同的结果，所有方法统一用其中一个对比就行
https://datascience.stackexchange.com/questions/126576/why-is-precision-recall-auc-different-from-average-precision-score
'''
def get_classify_metrics(pred_proba: np.array, labels: np.array, thres: float = 0.5):
    pred_labels = (pred_proba >= thres).astype(int)
    labels = (labels >= thres).astype(int) # 这行代码需要有，当labels是距离时，将距离转为标签

    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    mcc = matthews_corrcoef(labels, pred_labels)
    recall = recall_score(labels, pred_labels, zero_division=0)
    precision = precision_score(labels, pred_labels, zero_division=0)

    rocauc = roc_auc_score(labels, pred_proba)
    prauc = average_precision_score(labels, pred_proba)

    return acc, f1, mcc, recall, precision, rocauc, prauc

# TODO
def get_regression_metrics(pred_labels, labels):
    mse = mean_squared_error(labels, pred_labels)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, pred_labels)
    r2 = r2_score(labels, pred_labels)
    return mae, mse, rmse, r2

# # TODO 这个是MFPAD里的评价指标，需要再修改一下
# def get_regress2classify_metrics(pred_proba: np.array, labels: np.array):
#     preds_int = (preds >= 2).astype(int)
#     labels_int = (labels >= 2).astype(int)
#     acc = accuracy_score(labels_int, preds_int)
#     f1score = f1_score(labels_int, preds_int)
#     mcc = matthews_corrcoef(labels_int, preds_int)
#     rec = recall_score(labels_int, preds_int)
#     precision = precision_score(labels_int, preds_int)

#     return acc, f1score, mcc, rec, precision
# TODO
def set_random_seed():
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cv_train_and_test(dataset, cv_train_idx_list, cv_val_idx_list, cv_test_idx_list, args, evaluate_func):
    # 能做回归就一定能做分类，能做分类不一定能做回归
    class_metrics = ClassMetrics()
    regress_metrics = RegressMetrics()
    ret_joblib = {
        "real":[[] for _ in range(args.cv)],
        "pred":[[] for _ in range(args.cv)],
        "class_metric": None,
        "regress_metric": None,
    }

    for cv in tqdm.tqdm(range(args.cv)):
        logger.info("----- cv {} -----".format(cv))
        for fold in range(args.fold):
            cv_train_idx = cv_train_idx_list[cv][fold]
            cv_val_idx = cv_val_idx_list[cv][fold]
            cv_test_idx = cv_test_idx_list[cv][fold]
            y_pred = evaluate_func(dataset, cv_train_idx, cv_val_idx, cv_test_idx, args)
            update_metrics(class_metrics, regress_metrics, y_pred, dataset.subset(cv_test_idx).y, args)
            if args.label == "NHT_distance" or args.label == "AHT_distance":
                logger.info("mae: {:.3}, mse: {:.3}, rmse: {:.3}, r2: {:.3}".format(*regress_metrics.get_cur_metrics()))
            logger.info("acc: {:.3}, f1: {:.3}, mcc: {:.3}, recall: {:.3}, precision: {:.3}, rocauc: {:.3}, prauc: {:.3}".format(*class_metrics.get_cur_metrics()))
            ret_joblib["real"][cv].append(dataset.subset(cv_test_idx).y)
            ret_joblib["pred"][cv].append(y_pred)
    ret_joblib["class_metric"] = class_metrics
    ret_joblib["regress_metric"] = regress_metrics
    save_path = "result/auc/joblib/{}/{}/".format(args.method, args.label)
    save_file_path = save_path + "{}_{}.joblib".format(args.subtype, args.valid_type)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(ret_joblib, save_file_path)
    
    return class_metrics, regress_metrics


def backtest_train_and_test(dataset, backtest_train_idx_list, backtest_val_idx_list, backtest_test_idx_list, args, evaluate_func):
    class_metrics = ClassMetrics()
    regress_metrics = RegressMetrics()
    ret_joblib = {
        "real":[],
        "pred":[],
        "class_metric": None,
        "regress_metric": None,
    }

    # for cv in range(len(backtest_train_idx_list)):
    for cv in tqdm.tqdm(range(len(backtest_train_idx_list))):
        backest_train_idx = backtest_train_idx_list[cv]
        backest_val_idx = backtest_val_idx_list[cv]
        backest_test_idx = backtest_test_idx_list[cv]
        y_pred = evaluate_func(dataset, backest_train_idx, backest_val_idx, backest_test_idx, args)
        update_metrics(class_metrics, regress_metrics, y_pred, dataset.subset(backest_test_idx).y, args)
        if args.label == "NHT_distance" or args.label == "AHT_distance":
            logger.info("mae: {:.3}, mse: {:.3}, rmse: {:.3}, r2: {:.3}".format(*regress_metrics.get_cur_metrics()))
        logger.info("acc: {:.3}, f1: {:.3}, mcc: {:.3}, recall: {:.3}, precision: {:.3}, rocauc: {:.3}, prauc: {:.3}".format(*class_metrics.get_cur_metrics()))
        ret_joblib["real"].append(dataset.subset(backest_test_idx).y)
        ret_joblib["pred"].append(y_pred)

    ret_joblib["class_metric"] = class_metrics
    ret_joblib["regress_metric"] = regress_metrics
    save_path = "result/auc/joblib/{}/{}/".format(args.method, args.label)
    save_file_path = save_path + "{}_{}.joblib".format(args.subtype, args.valid_type)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(ret_joblib, save_file_path)

    return class_metrics, regress_metrics

def update_metrics(class_metrics, regress_metrics, y_pred, y, args):
    if args.label == "NHT_distance":
        regress_metrics.add_metric(*get_regression_metrics(y_pred, y))
        class_metrics.add_metric(*get_classify_metrics(y_pred, y, 2.0))
    elif args.label == "AHT_distance":
        regress_metrics.add_metric(*get_regression_metrics(y_pred, y))
        class_metrics.add_metric(*get_classify_metrics(y_pred, y, 2.0))
    elif args.label == "NHT_class" or args.label == "AHT_class":
        class_metrics.add_metric(*get_classify_metrics(y_pred, y, 0.5))
    else:
        assert(False)