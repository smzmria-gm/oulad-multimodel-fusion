# -*- coding: utf-8 -*-
"""生成论文四张配图（全部基于真实实验输出，不编造）。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np

FONT = 'SimHei'
plt.rcParams['font.sans-serif'] = [FONT]
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
EXP = os.path.join(ROOT, "results")

# =====================================================================
# 图1：多模型融合框架图（架构图，仅画已实证部分；LSTM/GCN 作未实证扩展虚框）
# =====================================================================
fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=200)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 12)
ax.axis('off')


def box(x, y, w, h, text, fc, ec, fs=10, weight='normal', text_color='#1a1a1a'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=weight, family=FONT, color=text_color)


def arrow(x1, y1, x2, y2, color='#555', style='-|>', ls='-', lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, color=color, lw=lw, linestyle=ls,
                 connectionstyle="arc3,rad=0.0"))


# OULAD 数据
box(5, 11.3, 3.4, 0.9, "OULAD 数据\n(高校选课注册 32 593 条)",
    fc='#FCE4D6', ec='#C55A11', fs=10, weight='bold')
# 特征工程（三源）
box(2, 9.6, 2.6, 0.85, "静态特征\n(人口统计/学分)", fc='#DDEBF7', ec='#2E75B6')
box(5, 9.6, 2.6, 0.85, "测评特征\n(参与/均分/首测)", fc='#DDEBF7', ec='#2E75B6')
box(8, 9.6, 2.6, 0.85, "VLE 行为特征\n(点击/活跃天)", fc='#DDEBF7', ec='#2E75B6')
# 基学习器（异质，5个）
bases = ["逻辑回归", "随机森林", "XGBoost", "LightGBM", "CatBoost"]
bx = [1, 3, 5, 7, 9]
for x, t in zip(bx, bases):
    box(x, 7.2, 1.7, 0.95, t, fc='#E2EFDA', ec='#548235')
# 融合层
box(3.5, 4.6, 2.8, 1.0, "Stacking\n(逻辑回归元学习器)", fc='#E4DFEC', ec='#6F42C1')
box(6.5, 4.6, 2.8, 1.0, "MLP 元学习器\n(可学习非线性组合)", fc='#E4DFEC', ec='#6F42C1')
# 输出
box(5, 2.6, 3.8, 0.95, "学业结果预测\n(二分类)", fc='#D6DCE4', ec='#1F3864', fs=10, weight='bold')

# 扩展接口（未实证）：移到基学习器层右侧，不接入主流程
ax.add_patch(FancyBboxPatch((11.25, 6.5), 2.2, 1.4,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             fc='#F2F2F2', ec='#808080', lw=1.5, linestyle=':'))
ax.text(12.35, 7.2, "框架扩展接口\n（本次未实证）\nLSTM / Transformer\nGCN / GAT",
        ha='center', va='center', fontsize=8.5, family=FONT, color='#595959')

# 连线（本次实证流程，实线）
arrow(5, 10.85, 2, 10.05); arrow(5, 10.85, 5, 10.05); arrow(5, 10.85, 8, 10.05)
arrow(2, 9.175, 2, 7.7); arrow(5, 9.175, 5, 7.7); arrow(8, 9.175, 8, 7.7)
for x in bx:
    arrow(x, 6.725, x, 6.05)
ax.plot([1, 9], [6.0, 6.0], color='#555', lw=1.6)
ax.text(5, 6.25, "Out-of-Fold 预测概率", ha='center', va='center',
        fontsize=8.5, family=FONT, color='#555')
arrow(3.5, 6.0, 3.5, 5.1); arrow(6.5, 6.0, 6.5, 5.1)
arrow(3.5, 4.1, 5, 3.08); arrow(6.5, 4.1, 5, 3.08)

# 未来扩展示意虚线（从扩展框指向基学习器层方向，标注为“可扩展”，不进入主流程）
arrow(11.25, 7.2, 9.85, 7.2, color='#808080', ls=':', lw=1.3)
ax.text(10.55, 7.58, "可扩展为\n基学习器", ha='center', va='center',
        fontsize=7.5, family=FONT, color='#808080')

# 右侧层级标注
for (x_, y_, lab) in [(10.0, 9.6, "特征工程"),
                      (10.0, 8.05, "基学习器（异质）"),
                      (10.0, 4.6, "融合层")]:
    ax.text(x_, y_, lab, ha='left', va='center', fontsize=9,
            family=FONT, color='#333', weight='bold')

# 底部图例：实线 = 本次实证；虚线 = 未来扩展方向
ax.plot([0.6, 1.4], [0.45, 0.45], color='#555', lw=1.6)
ax.text(1.6, 0.45, "本次实证流程", ha='left', va='center', fontsize=8.5,
        family=FONT, color='#555')
ax.plot([4.0, 4.8], [0.45, 0.45], color='#808080', lw=1.3, linestyle=':')
ax.text(5.0, 0.45, "未来扩展方向（本次未实证）", ha='left', va='center',
        fontsize=8.5, family=FONT, color='#808080')

plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_framework.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 图2：SHAP 全局特征重要性（水平条形图）
# =====================================================================
shap = pd.read_csv(os.path.join(EXP, "shap_top.csv"))
name_map = {
    "n_assess_attempted": "测评参与次数",
    "mean_score": "平均分",
    "vle_active_days": "VLE活跃天数",
    "first_score": "首次测评分",
    "mean_days_submitted": "平均提交延迟",
    "vle_total_clicks": "VLE总点击量",
    "highest_education_Lower Than A Level": "学历(低于A-Level)",
    "studied_credits": "注册学分",
    "early_score": "早期成绩",
    "date_registration": "注册日期",
}
shap = shap.head(10).iloc[::-1]
labels = [name_map.get(f, f) for f in shap["feature"]]
vals = shap["mean_abs_shap"].values

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
bars = ax.barh(labels, vals, color='#2E75B6')
for b, v in zip(bars, vals):
    ax.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va='center', fontsize=8.5, family=FONT)
ax.set_xlabel("平均绝对 SHAP 值", fontsize=10, family=FONT)
ax.set_title("图2  SHAP 全局特征重要性（前十位）", fontsize=11, family=FONT, weight='bold')
ax.tick_params(axis='y', labelsize=9)
for lbl in ax.get_yticklabels():
    lbl.set_family(FONT)
ax.set_xlim(0, max(vals) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_shap.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 图3：各模型 AUC 对比（基模型 + 融合策略）
# =====================================================================
base_names = ["逻辑回归", "随机森林", "XGBoost", "LightGBM", "CatBoost"]
base_auc = [0.9501, 0.9688, 0.9722, 0.9718, 0.9729]
fus_names = ["软投票", "静态加权", "Stacking", "MLP元学习器"]
fus_auc = [0.9716, 0.9716, 0.9727, 0.9727]

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
x = np.arange(len(base_names) + len(fus_names))
auc = base_auc + fus_auc
colors = ['#548235'] * 5 + ['#6F42C1'] * 4
bars = ax.bar(x, auc, color=colors, width=0.6)
ax.axhline(0.9730, color='#C55A11', ls='--', lw=1.3)
for b, v in zip(bars, auc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.4f}",
            ha='center', fontsize=8, family=FONT)
ax.set_xticks(x)
ax.set_xticklabels(base_names + fus_names, fontsize=9, family=FONT)
ax.set_ylim(0.94, 0.98)
ax.set_ylabel("AUC", fontsize=10, family=FONT)
ax.set_title("图3  各模型 AUC 对比（分层 5 折交叉验证）", fontsize=11, family=FONT, weight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#548235', label='基模型'),
                   Patch(color='#6F42C1', label='融合策略')],
          loc='lower right', fontsize=9, prop={'family': FONT})
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_auc.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 图4：消融实验（不同特征设置的融合 AUC）
# =====================================================================
abl_labels = ["仅静态特征", "静态+测评成绩", "全特征(含VLE)"]
abl_vals = [0.6565, 0.9696, 0.9727]

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
colors = ['#C00000', '#ED7D31', '#548235']
bars = ax.bar(abl_labels, abl_vals, color=colors, width=0.55)
for b, v in zip(bars, abl_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha='center', fontsize=10, family=FONT)
ax.set_ylim(0, 1.05)
ax.set_ylabel("融合 AUC", fontsize=10, family=FONT)
ax.set_title("图4  消融实验：不同特征设置的融合 AUC", fontsize=11, family=FONT, weight='bold')
ax.tick_params(axis='x', labelsize=9.5)
for lbl in ax.get_xticklabels():
    lbl.set_family(FONT)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_ablation.png"), dpi=200, bbox_inches='tight')
plt.close(fig)

print("FIGURES_DONE")
for f in ["fig1_framework.png", "fig2_shap.png", "fig3_auc.png", "fig4_ablation.png"]:
    p = os.path.join(FIG, f)
    print(f, round(os.path.getsize(p) / 1024, 1), "KB")
