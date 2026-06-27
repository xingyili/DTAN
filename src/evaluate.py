
import time

from src.model import *
from tqdm import tqdm
import torch.optim as optim
from src.utils import *
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from src.structure import *
import copy
from src.model import TransformerPairModel, TransformerModelConfig
# from src.model_ablation import *
# from model_tmp import ProteinPairDistanceModel, ModelConfig

# TODO 这里设置早停，还是和PREDAC-CNN一样的训练策略？？？PREDAC-CNN用测试集做验证？？？
def train(train_dataset, val_dataset, test_dataset, args):
    class_metrics = ClassMetrics()
    regress_metrics = RegressMetrics()

    cfg = TransformerModelConfig(
        in_channels=args.ae_dim,
        d_model=64,
        max_len=train_dataset.x.shape[3],
        enc_layers=2,
        enc_heads=4,
        enc_ff=256,
        dropout=0.1,
        task="regression",
    )
    # cfg = TransformerModelConfig(
    #     in_channels=args.ae_dim,
    #     d_model=128,
    #     max_len=train_dataset.x.shape[3],
    #     enc_layers=4,
    #     enc_heads=8,
    #     enc_ff=512,
    #     dropout=0.1,
    #     task="regression",
    # )
    model = TransformerPairModel(
            args,
            cfg,
            at_dim=train_dataset.at_name_encoded.shape[1],
            sr_dim=train_dataset.sr_name_encoded.shape[1]
        ).to(args.device)

    

    # model = HybridCNNMLP(328, train_dataset.at_name_encoded.shape[1] + train_dataset.sr_name_encoded.shape[1], 256).to(args.device)
    best_model = model
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss(beta=1.0)
    criterion = nn.L1Loss()
    mae_criterion = nn.L1Loss()
    # attn_reg_lambda = float(getattr(args, "attn_reg_lambda", 0.0))
    # attn_reg_lambda = 1e-6
    attn_reg_lambda = args.im_lambda
    lr = 1e-3
    base_decay = 1e-5
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=base_decay)  # 添加L2正则化

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7     # 最小学习率
    )
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    pb_miniters = 50
    pbar = tqdm.tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    best_val_loss = 1e6
    # train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        mae = 0
        for i, (images, labels, at_name_encoded, sr_name_encoded) in enumerate(train_loader):
            # labels = torch.eye(2)[labels.long()].to(args.device)
            #labels变为n*1
            labels = labels.float().unsqueeze(1).to(args.device)
            images = images.float().to(args.device)
            at_name_encoded = at_name_encoded.float().to(args.device)
            sr_name_encoded = sr_name_encoded.float().to(args.device)

            if attn_reg_lambda > 0 and getattr(args, "model", None) not in ("cnn", "no_encoder") and epoch < args.im_epoch:
                outputs, attn_info = model(images, at_name_encoded, sr_name_encoded, True)
                task_loss = criterion(outputs, labels)
                seq_len = images.shape[-1]
                # reg_loss = model.attention_regularization_loss(
                #     attn_info.get("incoming1_last"),
                #     attn_info.get("incoming2_last"),
                #     length=seq_len,
                # )
                reg_loss = model.get_attn_loss(
                    attn_info["incoming1_last"],
                    attn_info["incoming2_last"],
                    length=seq_len,
                )
                # print(attn_reg_lambda * reg_loss.item())
                loss = task_loss + attn_reg_lambda * reg_loss
            else:
                outputs = model(images, at_name_encoded, sr_name_encoded)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
            mae += mae_criterion(outputs, labels).item()

        epoch_loss /= len(train_loader)
        mae /= len(train_loader)
        
            
        model.eval()
        # 在验证集上查看性能
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for i, (images, labels, at_name_encoded, sr_name_encoded) in enumerate(val_loader):
            # for i, (images, labels, at_name_encoded, sr_name_encoded) in enumerate(test_loader):
                # labels = torch.eye(2)[labels.long()].to(args.device)
                labels = labels.float().unsqueeze(1).to(args.device)
                images = images.float().to(args.device)
                at_name_encoded = at_name_encoded.float().to(args.device)
                sr_name_encoded = sr_name_encoded.float().to(args.device)
                # at_name_encoded.zero_()  
                # sr_name_encoded.zero_() 
                outputs = model(images, at_name_encoded, sr_name_encoded)
                
                loss = criterion(outputs, labels)
                val_mae += mae_criterion(outputs, labels).item()
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            pbar.set_description(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train MAE: {mae:.4f}, Val MAE: {val_mae:.4f}')
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss

        # if (epoch+1)%25==0:
        
        if args.valid_type == "train":
            # y_pred = only_test(train_dataset, model, args, id=epoch)
            pass
        else:
            y_pred = test(test_dataset, model, args)
            update_metrics(class_metrics, regress_metrics, y_pred, test_dataset.y, args)
            logger.info("-------------------- epoch {}----------------------".format(epoch))
            logger.info(f"Train MAE: {mae:.4f}, Val MAE: {val_mae:.4f}")
            if args.label == "NHT_distance" or args.label == "AHT_distance":
                logger.info("mae: {:.3}, mse: {:.3}, rmse: {:.3}, r2: {:.3}".format(*regress_metrics.get_cur_metrics()))
            logger.info("acc: {:.3}, f1: {:.3}, mcc: {:.3}, recall: {:.3}, precision: {:.3}, rocauc: {:.3}, prauc: {:.3}".format(*class_metrics.get_cur_metrics()))


        # scheduler.step(val_loss)
        scheduler.step()
    return best_model
    # return model

def test(test_dataset, model, args):
    model.eval()
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False)
    all_attn = {}
    with torch.no_grad():
        for i, (images, _, at_name_encoded, sr_name_encoded) in enumerate(test_loader):
            images = images.float().to(args.device)
            at_name_encoded = at_name_encoded.float().to(args.device)
            sr_name_encoded = sr_name_encoded.float().to(args.device)
            outputs, attn = model(images, at_name_encoded, sr_name_encoded, True)
            if i == 0:
                y_pred = outputs
                # all_attn["incoming1_last"] = attn["incoming1_last"]
                # all_attn["incoming2_last"] = attn["incoming2_last"]
            else:
                y_pred = torch.cat((y_pred, outputs), dim=0)
                # all_attn["incoming1_last"] = torch.cat((all_attn["incoming1_last"], attn["incoming1_last"]), dim=0)
                # all_attn["incoming2_last"] = torch.cat((all_attn["incoming2_last"], attn["incoming2_last"]), dim=0)
    # cur_day = time.strftime("%Y-%m-%d")
    # cur_time = time.strftime("%H-%M-%S")
    # joblib.dump(all_attn, "result/importance/attn_{}_{}.joblib".format(args.subtype, cur_day + "_" + cur_time))
    return y_pred.cpu().detach().numpy()
    # return y_pred.cpu().detach().numpy(), all_attn.cpu().detach().mean(dim=0).numpy()

    # a["incoming1_last"].mean(dim=0)
    # 


def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_dataset = dataset.subset(train_idx)
    train_dataset.fit_ohe()
    val_dataset = dataset.subset(val_idx)
    test_dataset = dataset.subset(test_idx)

    val_dataset.at_ohe = train_dataset.at_ohe
    val_dataset.sr_ohe = train_dataset.sr_ohe
    test_dataset.at_ohe = train_dataset.at_ohe
    test_dataset.sr_ohe = train_dataset.sr_ohe
    val_dataset.transform_ohe()
    test_dataset.transform_ohe()

    optied_model = train(train_dataset, val_dataset, test_dataset, args)
    return test(test_dataset, optied_model, args)