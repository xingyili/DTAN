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

# 这个方法只能用在H3数据上，因为构建特征用到的生物学特性只有H3的
parser = argparse.ArgumentParser(description="示例命令行参数程序")
parser.add_argument("--percentage", type=float)
parser.add_argument("--low", type=str, choices=["ae", "pca", "none"], help="是否使用自编码器进行特征编码")
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
parser.add_argument("--model", type=str, default="full", choices=["full", "cnn", "no_encoder"], help="模型类型")
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
args.ae_dim = args.ae_dim if args.low in ["ae", "pca"] else 566

cur_day = time.strftime("%Y-%m-%d")
cur_time = time.strftime("%H-%M-%S")
# print(cur_day)
print(cur_time)
# exit()
logger.remove() # 禁止输出到控制台
logger.add("log/{}/{}/{}/{}_{}.log".format(cur_day, args.model, args.subtype, args.valid_type, cur_time), level="DEBUG", mode="w", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

args2log(args)
if args.device == "auto":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(args.device)


logger.info("loading data...")
at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, _, _ = load_data(args)




n = len(at_seqs)
k = (int)(n*args.percentage)
logger.info("The number of samples is {}".format(k))
idx = random.sample(range(n), k)
at_seqs_s  = [at_seqs[i]  for i in idx]
sr_seqs_s  = [sr_seqs[i]  for i in idx]
labels_s   = labels[idx]  
at_year_s  = [at_year[i]  for i in idx]
sr_year_s  = [sr_year[i]  for i in idx]
at_name_s  = [at_name[i]  for i in idx]
sr_name_s  = [sr_name[i]  for i in idx]
at_seqs[:]  = at_seqs_s
sr_seqs[:]  = sr_seqs_s
labels = labels_s
at_year[:]  = at_year_s
sr_year[:]  = sr_year_s
at_name[:]  = at_name_s
sr_name[:]  = sr_name_s


# %% 构建特征
logger.info("construct features...")
feature_constructor = AeFeature(args)
features = feature_constructor.construct_feature(at_seqs, sr_seqs)
dataset = FluDataset(features, labels, at_name, sr_name)
logger.info("The shape of features is {}".format(features.shape))



cv_train_idx_list, cv_val_idx_list, cv_test_idx_list = load_cv_idx(features.shape[0], args)
train_idx = cv_train_idx_list[0][0] + cv_test_idx_list[0][0]
val_idx = cv_val_idx_list[0][0]
start_time = time.time()
opti_model, at_ohe, sr_ohe = only_train(dataset, cv_train_idx_list, cv_val_idx_list, cv_test_idx_list, args, evaluate_func)
end_time = time.time()
logger.info("training time is : {}".format(end_time - start_time))


# 保存模型
if args.percentage == 1.0 and args.low == "ae":
    model_save_path = "saved_models/{}/".format(args.subtype)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_file_path = model_save_path + "{}_{}.pt".format(args.subtype, args.low)
    torch.save(opti_model.state_dict(), model_file_path)
    joblib.dump(at_ohe, model_save_path + "{}_{}_at_ohe.joblib".format(args.subtype, args.low))
    joblib.dump(sr_ohe, model_save_path + "{}_{}_sr_ohe.joblib".format(args.subtype, args.low))