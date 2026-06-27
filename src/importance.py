import pandas as pd
import joblib
# 下标从1开始
influenza_sites = {
    "H3": {
        "Key sites": [145, 155, 156, 158, 159, 189, 193],
        "Epitope A": [122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146],
        "Epitope B": [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198],
        "Epitope E": [62, 63, 64, 65, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 260, 261, 262, 263, 264, 265],
        "Epitope C": [273, 274, 275, 276, 277, 278, 279, 280, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        "Epitope D": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219],
        "130-loop": [134, 135, 136, 137, 138],
        "190-helix": [190, 191, 192, 193, 194, 195, 196, 197, 198],
        "220-loop": [221, 222, 223, 224, 225, 226, 227, 228]
    },
    "H1": {
        "Key sites": [127, 141, 153, 154, 155, 156],
        "Epitope A": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
        "Epitope B": [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170],
        "Epitope E": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 257, 258, 259, 260, 261, 262, 263],
        "Epitope C": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 271, 272, 273, 274, 275, 276, 277, 278],
        "Epitope D": [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
        "130-loop": [131, 132, 133, 134, 135],
        "190-helix": [187, 188, 189, 190, 191, 192, 193, 194, 195],
        "220-loop": [218, 219, 220, 221, 222, 223, 224, 225]
    },
    "H5": {
        "Key sites": [119, 123, 125, 126, 127, 129, 138, 140, 141, 151, 152, 153, 154, 155, 156, 185, 189],
        "Epitope A": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
        "Epitope B": [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194],
        "Epitope E": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 256, 257, 258, 259, 260, 261, 262],
        "Epitope C": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 270, 271, 272, 273, 274, 275, 276, 277],
        "Epitope D": [197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
        "130-loop": [130, 131, 132, 133, 134],
        "190-helix": [186, 187, 188, 189, 190, 191, 192, 193, 194],
        "220-loop": [217, 218, 219, 220, 221, 222, 223, 224]
    }
}



# site_categories_reversed = {1: 'Key sitess', 2: 'epitopes_A', 3: 'epitopes_B', 4: 'epitopes_E', 5: 'epitopes_C', 6: 'epitopes_D', 7: '130-loop', 8: '190-helix', 9: '220-loop', 0: 'Glycosylation'}
# {site:{site所属的重要区域类别}}
# site_to_label = {145: {1, 2}, 155: {1, 3}, 156: {1, 3}}



# def enrichment_analysis_multilabel(importance_file, subtype, top_k=20, gly_sites = None):
#     '''
#     important_file:.csv, 必须包含两列['position', 'importance']
#     subtype
#     top_k: 前k个重要位点
#     '''
#     from scipy.stats import fisher_exact
#     from statsmodels.stats.multitest import multipletests
#     from utils.utils import get_important_sites_with_multilabel

#     # 1. 模型预测的各个位点的重要性表
#     df = pd.read_csv(importance_file)
#     background = set(df['position'].astype(int).values)
#     topk = set(df.sort_values('importance', ascending=False)
#                  .head(top_k)['position'].astype(int).values)

#     # 2. 获取 position -> set(labels) 的映射
#     site_to_labels = get_important_sites_with_multilabel(subtype)
#     # {site:{site所属的重要区域类别}}
#     # site_to_label = {145: {1, 2}, 155: {1, 3}, 156: {1, 3}}

#     # 3. 构造 category -> positions 的反向映射
#     from collections import defaultdict
#     cat_to_sites = defaultdict(set)
#     for pos, labels in site_to_labels.items():
#         for label in labels:
#             cat_to_sites[label].add(pos)
    
#     if gly_sites is not None:
#         site_categories_reversed[0] = 'Glycosylation'
#         cat_to_sites[0] = set(gly_sites)
#     results = []
#     for cat, cat_set in cat_to_sites.items():
#         a = len(topk & cat_set)
#         site = topk & cat_set
#         b = len(topk) - a
#         c = len((background & cat_set)) - a
#         d = len(background) - (a + b + c)
#         oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
#         results.append({
#             'category': site_categories_reversed[int(cat)],
#             'in_topk': a,
#             'cat_size': len(cat_set),
#             'pvalue': pvalue,
#             'oddsratio': oddsratio,
#             'site':site
#         })
#     res_df = pd.DataFrame(results).sort_values('pvalue')
#     res_df['p_adj'], res_df['reject'] = multipletests(res_df['pvalue'], method='fdr_bh')[:2]
#     return res_df

def plot_site_importance_line_and_point_multi(
        position_importance_dict, 
        region_sites_dict, 
        color_map, 
        enrich_df, 
        draw_order, 
        save_path=None,
        subtype=None,
    ):
    """
    绘制 site importance 折线+散点图。
    
    参数:
    - position_importance_dict: dict, {pos(int): importance(float)}
    - region_sites_dict: dict, {region_name(str): [site1, site2, ...]}
    - color_map: dict, {region_name(str): hex_color(str)}
    - enrich_df: DataFrame, 包含 'category', 'pvalue', 'site' 列
    - draw_order: list, region_name 的绘制和图例顺序
    - save_path: str, 保存路径
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # === 0. 全局绘图风格设置 ===
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 16,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 14,
        'font.family': 'Times New Roman'
    })

    # # === 1. 准备基础数据 ===
    # # 提取在 Top K 中的位点 (用于散点图着色：黑色为Top，灰色为普通)
    # top_sites = set()
    # if 'site' in enrich_df.columns:
    #     for s in enrich_df["site"]:
    #         try:
    #             # 兼容处理集合的字符串表示或直接集合
    #             val = eval(str(s)) if isinstance(s, str) else s
    #             top_sites |= set(val)
    #         except:
    #             continue

    # 将 importance 字典转为 DataFrame 并排序
    positions = list(position_importance_dict.keys())
    importances = [position_importance_dict[p] for p in positions]
    df = pd.DataFrame({"position": positions, "importance": importances})
    df = df.sort_values("position").reset_index(drop=True)

    # === 2. 绘图布局（单轴） ===
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 4))
    # fig, ax1 = plt.subplots(1, 1, )

    
    
    # 动态设置 Y 轴范围：按最小值/最大值上下各留 0.1
    ymin = float(df["importance"].min())
    ymax = float(df["importance"].max())
    ax1.set_ylim(ymin - 0.5, ymax + 0.5)

    # === 5. 绘制区域竖线 (核心逻辑) ===
    # 将 DataFrame 的 position 转为 set 加速查找
    valid_positions = set(df["position"].values)
    
    handles, legend_labels = [], []

    for region_name in draw_order:
        # 获取该区域包含的位点
        sites_in_region = region_sites_dict.get(region_name, [])
        color = color_map.get(region_name, "#CCCCCC")
        
        # 获取 P-value 用于图例
        # 注意：enrich_df 中的 category 名称必须与 region_name 一致


        # pval_row = enrich_df[enrich_df["category"] == region_name]
        
        # 构造图例标签
        label_text = region_name
        # if not pval_row.empty:
        #     pval = pval_row["pvalue"].values[0]
        #     if pval < 0.05:
        #         label_text = f"{region_name} (p<0.05)"
        #     else:
        #         label_text = f"{region_name} (p={pval:.3g})"
        # else:
        #     label_text = f"{region_name} (p=NA)"



        # 绘制竖线
        # 只有当该区域的位点确实存在于 importance 数据中时才画线
        lines_drawn = False
        line_handle = None
        
        for pos in sites_in_region:
            if pos in valid_positions:
                h = ax1.axvline(pos, color=color, alpha=1.0, linewidth=2)
                lines_drawn = True
                line_handle = h # 保存句柄用于图例

        # # 如果这个区域画了线，就添加到图例中
        if lines_drawn and line_handle is not None:
            handles.append(line_handle)
            legend_labels.append(label_text)

    # === 3. 上半部分：折线图 ===
    ax1.plot(df["position"], df["importance"], color="black", linewidth=1.5)
    ax1.set_ylabel("Importance Score", fontsize=20, fontweight="bold")
    ax1.set_xlabel("Site Position", fontsize=20, fontweight="bold")

    # === 6. 添加图例并保存 ===
    if handles and legend_labels:
        # ax1.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(1, 0.5))
        leg = fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.1),  # 向下移动
            ncol=5,     # 横向排列
            fontsize=18,          # 字体变大
            handlelength=2.0,
            frameon=False
        )
        for line in leg.get_lines():
            line.set_linewidth(5)

    plt.tight_layout()
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=600, bbox_inches='tight') # bbox_inches='tight' 防止图例被切掉
    plt.close()

def report_topk_and_regions(position_importance_dict, region_sites_dict, draw_order, top_k=20, title=""):
    # 1) top-k 位点（按重要性从高到低；同分按位点号）
    topk_sorted = sorted(
        position_importance_dict.items(),
        key=lambda kv: (-float(kv[1]), int(kv[0]))
    )[:top_k]
    topk_positions = [int(p) for p, _ in topk_sorted]
    topk_set = set(topk_positions)

    print("\n" + "=" * 80)
    if title:
        print(title)
    print(f"Top-{top_k} sites (position: importance):")
    # for p, v in topk_sorted:
    #     print(f"  {int(p)}: {float(v):.6g}")

    # 2) 计算已知区域合集（用于 unknown）
    known_union = set()
    for region_name in draw_order:
        known_union |= set(region_sites_dict.get(region_name, []))

    # 3) 每个区域包含哪些 top-k 位点 + unknown
    print(f"\nTop-{top_k} sites in each region:")
    for region_name in draw_order:
        region_sites = set(region_sites_dict.get(region_name, []))
        hits = sorted(topk_set & region_sites)
        print(f"  - {region_name}: {hits} (n={len(hits)})")

    unknown_hits = sorted(topk_set - known_union)
    print(f"  - unknown: {unknown_hits} (n={len(unknown_hits)})")

    return topk_positions

subtype = 'H1'
# 从joblib文件加载注意力分数数据
attn_data = joblib.load(f"result/importance/2026-03-04/H1_19:45:44/0.joblib")
# attn_data = joblib.load(f"result/importance/2026-03-04/H3_19:36:02/0.joblib")
# attn_data = joblib.load(f"result/importance/2026-03-04/H5_19:25:30/0.joblib")
# 提取最后一层的注意力分数
incoming1_last_attn = attn_data["incoming1_last"] # shape: (num_samples, num_heads, seq_len)
incoming2_last_attn = attn_data["incoming2_last"] # shape: (num_samples, num_heads, seq_len)
# 计算每个位点的平均注意力分数（先对样本和头进行平均）
# pos_importance_incoming1 = incoming1_last_attn.mean(dim=0) # shape: (seq_len,)
# pos_importance_incoming2 = incoming2_last_attn.mean(dim=0) # shape: (seq_len,)
pos_importance_incoming1 = incoming1_last_attn
pos_importance_incoming2 = incoming2_last_attn
print(pos_importance_incoming1.shape)
pos_im_ag_dict = {}
pos_im_sr_dict = {}
for i in range(pos_importance_incoming1.shape[0]):
    pos_im_ag_dict[i+1] = pos_importance_incoming1[i].item()
    pos_im_sr_dict[i+1] = pos_importance_incoming2[i].item()

color_map = {
    "Epitope A": "#FAEEED",
    "Epitope B": "#C9E0FC",
    "Epitope E": "#D9C7E3",
    "Epitope C": "#D4F2DE",
    "Epitope D": "#F5DEDC",
    "130-loop": "#CDECE6",
    "190-helix": "#D1F8F5",
    "220-loop": "#FAF3D4",
    "Key sites": "#F4DECA",
}

plot_site_importance_line_and_point_multi(pos_im_ag_dict, influenza_sites[subtype], color_map=color_map, enrich_df=None, save_path=f"result/figure/site_importance_{subtype}_ag.png", draw_order=["Key sites", "Epitope A", "Epitope B", "Epitope E", "Epitope C", "Epitope D", "130-loop", "190-helix", "220-loop"])
plot_site_importance_line_and_point_multi(pos_im_sr_dict, influenza_sites[subtype], color_map=color_map, enrich_df=None, save_path=f"result/figure/site_importance_{subtype}_sr.png", draw_order=["Key sites", "Epitope A", "Epitope B", "Epitope E", "Epitope C", "Epitope D", "130-loop", "190-helix", "220-loop"])

# F5DEDC Epitope D
# D4F2DE Epitope C
# C9E0FC Epitope B
# F4DECA Key sites
# FAEEED Epitope A
# F8D2D1 糖基化位点
# D9C7E3 Epitope E
# CDECE6 130-loop
# D1F8F5 190-helix
# FAF3D4 220-loop
# {
#     "Epitope A": "#FAEEED",
#     "Epitope B": "#C9E0FC",
#     "Epitope E": "#D9C7E3",
#     "Epitope C": "#D4F2DE",
#     "Epitope D": "#F5DEDC",
#     "130-loop": "#D1F8F5",
#     "190-helix": "#D1F8F5",
#     "220-loop": "#FAF3D4",
# }

# 3. antigenic_epitopes_： 五个抗原表位（ one~five分别对应抗原表位ABECD，H1 额外加的区间归入抗原表位B）

top_k = 30
topk_ag = report_topk_and_regions(
    pos_im_ag_dict,
    influenza_sites[subtype],
    draw_order=["Key sites", "Epitope A", "Epitope B", "Epitope E",
               "Epitope C", "Epitope D", "130-loop", "190-helix", "220-loop"],
    top_k=top_k,
    title=f"[{subtype}] AG"
)

topk_sr = report_topk_and_regions(
    pos_im_sr_dict,
    influenza_sites[subtype],
    draw_order=["Key sites", "Epitope A", "Epitope B", "Epitope E",
               "Epitope C", "Epitope D", "130-loop", "190-helix", "220-loop"],
    top_k=top_k,
    title=f"[{subtype}] SR"
)