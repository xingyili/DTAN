# %%
import os
import sys
from src.dataloader import *
from src.utils import *
from src.feature import *
from src.evaluate import *
from loguru import logger
import argparse
import time

parser = argparse.ArgumentParser(description="示例命令行参数程序")
parser.add_argument("--low", type=str, choices=["ae", "pca", "none", "onehot"], help="是否使用自编码器进行特征编码")
parser.add_argument("--ae_dim", type=int, default=64, help="自编码器编码维度")
parser.add_argument("--cv", type=int, default=5, help="交叉验证次数")
parser.add_argument("--fold", type=int, default=5, help="交叉验证折数")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--n_jobs", type=int, default=os.cpu_count(), help="并行运算的核数")
parser.add_argument("--seq_path", type=str, default="dataset0722tmp/H1N1_N2_HA_syed.csv", help="序列数据的路径", required=True)
parser.add_argument("--anti_path", type=str, default="dataset0722tmp/H1N1_N2_HI_syed.csv", help="抗原性数据的路径", required=True)
parser.add_argument("--subtype", type=str, choices=["H1", "H3", "H5"], help="数据亚型", required=True)
parser.add_argument("--label", type=str, default="NHT_class", choices=["NHT_class", "NHT_distance", "AHT_class", "AHT_distance"], help="使用`距离`/`变异情况`作为标签", required=True)
parser.add_argument("--valid_type", type=str, choices=["titer", "virus", "time", "train", "test"], help="交叉验证划分类型", required=True)
# parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--model", type=str, default="full", choices=["full", "cnn", "no_encoder", "no_meta"], help="模型类型")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--device", type=str, default="cuda:5", help="训练设备")
parser.add_argument("--time_steps", type=int, default=5, help="时间序列滑动次数")
parser.add_argument("--val_window", type=int, help="时间测试的验证窗口")
parser.add_argument("--test_window", type=int, help="时间测试的测试窗口")
parser.add_argument("--other", type=str, help="额外信息")
parser.add_argument("--test_file", type=str, help="测试文件路径")
parser.add_argument("--im_lambda", type=float, default=0, help="特征重要性正则化强度")
parser.add_argument("--im_epoch", type=int, default=0, help="特征重要性预训练先验")
args = parser.parse_args()
args.method = "flux"
args.n_jobs = 32
args.ae_dim = args.ae_dim if args.low in ["ae", "pca", "onehot"] else 566

cur_day = time.strftime("%Y-%m-%d")
cur_time = time.strftime("%H-%M-%S")
# print(cur_day)
print(cur_time)
# exit()
logger.remove() # 禁止输出到控制台
logger.add("log/{}/{}_{}/{}/{}_{}.log".format(cur_day, args.model, args.low, args.subtype, args.valid_type, cur_time), level="DEBUG", mode="w", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

args2log(args)
if args.device == "auto":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(args.device)
args.cur_time = cur_time
args.cur_day = cur_day
if args.valid_type in ["titer", "virus", "time", "train"]:
# %% 加载数据
    logger.info("loading data...")
    at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, _, _ = load_data(args)

    # %% 构建特征
    logger.info("construct features...")
    feature_constructor = AeFeature(args)
    features = feature_constructor.construct_feature(at_seqs, sr_seqs)
    dataset = FluDataset(features, labels, at_name, sr_name)
if args.valid_type == "train":
    cv_train_idx_list, cv_val_idx_list, cv_test_idx_list = load_cv_idx(features.shape[0], args)
    opti_model, at_ohe, sr_ohe = only_train(dataset, cv_train_idx_list, cv_val_idx_list, cv_test_idx_list, args, evaluate_func)
    # # 保存模型

    # 如果需要保存模型，就保留下面的代码
    model_save_path = "saved_models/{}/".format(args.subtype)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_file_path = model_save_path + "{}.pt".format(args.subtype)
    torch.save(opti_model.state_dict(), model_file_path)
    joblib.dump(at_ohe, model_save_path + "{}_at_ohe.joblib".format(args.subtype))
    joblib.dump(sr_ohe, model_save_path + "{}_sr_ohe.joblib".format(args.subtype))

elif args.valid_type == "test":
    # 加载模型
    model_load_path = "saved_models/{}/".format(args.subtype)
    model_file_path = model_load_path + "{}.pt".format(args.subtype)
    at_ohe = joblib.load(model_load_path + "{}_at_ohe.joblib".format(args.subtype))
    sr_ohe = joblib.load(model_load_path + "{}_sr_ohe.joblib".format(args.subtype))
    if args.subtype == "H1":
        opti_model = TransformerPairModel(args, TransformerModelConfig(max_len=326), at_dim=len(at_ohe.get_feature_names_out()), sr_dim=len(sr_ohe.get_feature_names_out()))
    elif args.subtype == "H3":
        opti_model = TransformerPairModel(args, TransformerModelConfig(max_len=328), at_dim=len(at_ohe.get_feature_names_out()), sr_dim=len(sr_ohe.get_feature_names_out()))
    elif args.subtype == "H5":
        opti_model = TransformerPairModel(args, TransformerModelConfig(max_len=317), at_dim=len(at_ohe.get_feature_names_out()), sr_dim=len(sr_ohe.get_feature_names_out()))
    opti_model.load_state_dict(torch.load(model_file_path))
    opti_model = opti_model.to(args.device)
    logger.info("----------- test on {} -----------".format(args.test_file))
    test_seq_df = pd.read_csv("download_ret/{}_with_seq.csv".format(args.subtype))
    test_at_seqs = test_seq_df["seq1"].tolist()
    test_sr_seqs = test_seq_df["seq2"].tolist()
    test_at_names = test_seq_df["virus1"].tolist()
    test_sr_names = test_seq_df["virus2"].tolist()
    feature_constructor = AeFeature(args)
    test_features = feature_constructor.construct_feature(test_at_seqs, test_sr_seqs)
    test_dataset = FluDataset(test_features, None, test_at_names, test_sr_names)
    test_dataset.at_name_encoded = at_ohe.transform(np.array(test_dataset.at_name).reshape(-1,1))
    test_dataset.sr_name_encoded = sr_ohe.transform(np.array(test_dataset.sr_name).reshape(-1,1))
    y_pred = test( test_dataset,opti_model, args)
    print(y_pred)
    ret = pd.DataFrame({"virus1": test_at_names, "virus2": test_sr_names, "antigenic": y_pred.reshape(-1), "titer_type": ["NORMAL"]*len(y_pred)})
    ret.to_csv("download_ret/{}_results.csv".format(args.subtype), index=False)

    
elif args.valid_type == "titer":
    # %% 交叉验证
    cv_train_idx_list, cv_val_idx_list, cv_test_idx_list = load_cv_idx(features.shape[0], args)
    logger.info("----------- cross valid... -----------")
    cv_class_metrics, cv_regress_metrics = cv_train_and_test(dataset, cv_train_idx_list, cv_val_idx_list, cv_test_idx_list, args, evaluate_func)
    logger.info("----------- mean -----------")
    logger.info("acc: {:.3}, f1: {:.3}, mcc: {:.3}, recall: {:.3}, precision: {:.3}, rocauc: {:.3}, prauc: {:.3}".format(*cv_class_metrics.get_ave_metrics()))
    if args.label == "NHT_distance" or args.label == "AHT_distance":
        logger.info("mae: {:.3}, mse: {:.3}, rmse: {:.3}, r2: {:.3}".format(*cv_regress_metrics.get_ave_metrics()))


