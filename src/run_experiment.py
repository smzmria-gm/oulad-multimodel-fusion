# -*- coding: utf-8 -*-
"""
OULAD 高校学业成绩预测 — 多模型融合实验管线
- 数据单元: 一次选课注册 (code_module, code_presentation, id_student)
- 目标: final_result 二分类 (Pass/Distinction=1, Fail/Withdrawn=0)
- 基模型: 逻辑回归 + 随机森林 + XGBoost + LightGBM + CatBoost (异质)
- 融合: 软投票 / Stacking(LogisticRegression) / 注意力融合(MLP 元学习器)
- 评估: 5折分层CV, OOF 防泄漏; Acc/F1/AUC/Recall/Precision; 消融; SHAP
"""
import os, time, warnings, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             recall_score, precision_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)
DATA = os.environ.get("OULAD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.environ.get("EXP_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)
USE_VLE = True

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---------- 1. 读取核心表 ----------
log("Loading studentInfo ...")
si = pd.read_csv(os.path.join(DATA, "studentInfo.csv"))
log("studentInfo shape:", si.shape, "classes:", si['final_result'].value_counts().to_dict())

# 目标二分类
pos = {'Pass', 'Distinction'}
si['y'] = si['final_result'].apply(lambda x: 1 if x in pos else 0)
log("Positive(Pass/Distinction):", int(si['y'].sum()), " Negative:", int((1-si['y']).sum()))

keys = ['code_module', 'code_presentation', 'id_student']

# ---------- 2. 静态特征 (来自 studentInfo) ----------
static_cols = ['gender', 'region', 'highest_education', 'imd_band',
               'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability']
df = si[keys + ['y']].copy()
static = si[keys + static_cols].copy()
# imd_band 缺失填充
static['imd_band'] = static['imd_band'].fillna('Missing')
df = df.merge(static, on=keys, how='left')

# ---------- 3. 注册时间特征 ----------
log("Loading studentRegistration ...")
reg = pd.read_csv(os.path.join(DATA, "studentRegistration.csv"))
reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
reg['date_registration'] = reg['date_registration'].fillna(reg['date_registration'].median())
df = df.merge(reg[keys + ['date_registration']], on=keys, how='left')

# ---------- 4. 测评特征 (assessments + studentAssessment) ----------
log("Loading assessments + studentAssessment ...")
ass = pd.read_csv(os.path.join(DATA, "assessments.csv"))
sa = pd.read_csv(os.path.join(DATA, "studentAssessment.csv"))
sa = sa.merge(ass[['id_assessment', 'code_module', 'code_presentation', 'date', 'assessment_type']],
              on='id_assessment', how='left')
sa['score'] = pd.to_numeric(sa['score'], errors='coerce')
sa['date_submitted'] = pd.to_numeric(sa['date_submitted'], errors='coerce')
sa['date'] = pd.to_numeric(sa['date'], errors='coerce')
# 每个注册单元聚合
sa_g = sa.groupby(keys).agg(
    n_assess_attempted=('score', 'count'),
    mean_score=('score', 'mean'),
    first_score=('score', 'first'),
    mean_days_submitted=('date_submitted', lambda s: (s - sa.loc[s.index, 'date']).mean()),
).reset_index()
# 前两次测评均值 (早期预测代理)
sa_sorted = sa.sort_values(['code_module', 'code_presentation', 'id_student', 'date'])
sa_sorted['rn'] = sa_sorted.groupby(keys).cumcount()
early = sa_sorted[sa_sorted['rn'] < 2].groupby(keys)['score'].mean().reset_index(name='early_score')
sa_g = sa_g.merge(early, on=keys, how='left')
df = df.merge(sa_g, on=keys, how='left')

# ---------- 5. VLE 行为特征 (分块聚合, 432MB) ----------
if USE_VLE:
    log("Aggregating VLE (chunked, 432MB) ...")
    try:
        vle_path = os.path.join(DATA, "studentVle.csv")
        agg = {}  # key tuple -> [total_clicks, set(dates)]
        total_rows = 0
        for chunk in pd.read_csv(vle_path, usecols=['code_module', 'code_presentation', 'id_student', 'date', 'sum_click'],
                                 dtype={'code_module': 'category', 'code_presentation': 'category',
                                        'id_student': 'int32', 'date': 'int16', 'sum_click': 'int32'},
                                 chunksize=2_000_000):
            total_rows += len(chunk)
            g = chunk.groupby(keys, observed=True).agg(
                clicks=('sum_click', 'sum'), ndays=('date', 'nunique')).reset_index()
            for _, r in g.iterrows():
                k = (r['code_module'], r['code_presentation'], int(r['id_student']))
                if k in agg:
                    agg[k][0] += r['clicks']; agg[k][1] += r['ndays']
                else:
                    agg[k] = [r['clicks'], r['ndays']]
        log("VLE rows scanned:", total_rows, " unique enrollment units:", len(agg))
        vle_df = pd.DataFrame([(k[0], k[1], k[2], v[0], v[1]) for k, v in agg.items()],
                              columns=keys + ['vle_total_clicks', 'vle_active_days'])
        df = df.merge(vle_df, on=keys, how='left')
        df['vle_total_clicks'] = df['vle_total_clicks'].fillna(0)
        df['vle_active_days'] = df['vle_active_days'].fillna(0)
        log("VLE features merged. vle_total_clicks mean:", round(df['vle_total_clicks'].mean(), 1))
    except Exception as e:
        log("VLE aggregation failed, continue without VLE:", e)
        USE_VLE = False

# ---------- 6. 特征工程收尾 ----------
feat_num = ['date_registration', 'num_of_prev_attempts', 'studied_credits',
            'n_assess_attempted', 'mean_score', 'first_score', 'early_score', 'mean_days_submitted']
if USE_VLE:
    feat_num += ['vle_total_clicks', 'vle_active_days']
for c in feat_num:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

cat_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
X_num = df[feat_num].astype(float).values
X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=False).astype(float).values
X = np.hstack([X_num, X_cat])
y = df['y'].values
log("Feature matrix:", X.shape, " positivity rate:", round(y.mean(), 3))

# 特征名
feat_names = feat_num + list(pd.get_dummies(df[cat_cols].astype(str)).columns)

# ---------- 7. 基模型定义 ----------
def make_models():
    return {
        'LogisticRegression': Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None,
                                              class_weight='balanced', n_jobs=-1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9,
                                     eval_metric='logloss', n_jobs=-1, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                      subsample=0.9, colsample_bytree=0.9,
                                      class_weight='balanced', n_jobs=-1, random_state=42,
                                      verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                       auto_class_weights='Balanced', random_seed=42,
                                       verbose=False),
    }

models = make_models()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 8. 训练基模型, 收集 OOF 概率 (防泄漏) ----------
oof = {}        # name -> (n,) proba of class 1
fold_metrics = []  # 记录每个模型每折指标
log("Training base models with 5-fold OOF ...")
for name, mdl in models.items():
    log("  ->", name)
    proba = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[va])[:, 1]
        proba[va] = p
        # 该折验证指标
        pred = (p >= 0.5).astype(int)
        fold_metrics.append({
            'model': name, 'fold': fold + 1,
            'accuracy': accuracy_score(y[va], pred),
            'f1': f1_score(y[va], pred, zero_division=0),
            'auc': roc_auc_score(y[va], p),
            'recall': recall_score(y[va], pred, zero_division=0),
            'precision': precision_score(y[va], pred, zero_division=0),
        })
    oof[name] = proba

oof_df = pd.DataFrame(oof)
pd.DataFrame(fold_metrics).to_csv(os.path.join(OUT, "base_fold_metrics.csv"), index=False)
log("Base OOF done. Base AUCs:",
    {n: round(roc_auc_score(y, oof[n]), 4) for n in models})

# ---------- 9. 融合策略 ----------
def evaluate_pred(p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p), recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0))

# (a) 软投票
vote_p = oof_df.mean(axis=1).values
fusion_results = {'SoftVoting': evaluate_pred(vote_p)}

# (b) Stacking: OOF概率 -> LogisticRegression 元学习器, 用5折交叉验证评估
meta_X = oof_df.values
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_oof = np.zeros(len(y))
for tr, va in skf2.split(meta_X, y):
    mr = LogisticRegression(max_iter=2000, class_weight='balanced')
    mr.fit(meta_X[tr], y[tr])
    stack_oof[va] = mr.predict_proba(meta_X[va])[:, 1]
fusion_results['Stacking(LR)'] = evaluate_pred(stack_oof)

# (c) 注意力融合: MLP 元学习器 (可学习非线性组合)
mlp_oof = np.zeros(len(y))
for tr, va in skf2.split(meta_X, y):
    mp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                       alpha=1e-3, random_state=42)
    mp.fit(meta_X[tr], y[tr])
    mlp_oof[va] = mp.predict_proba(meta_X[va])[:, 1]
fusion_results['AttentionFusion(MLP)'] = evaluate_pred(mlp_oof)

# (d) 静态性能加权注意力 (softmax over base AUC)
base_auc = np.array([roc_auc_score(y, oof[n]) for n in models])
w = np.exp(base_auc) / np.exp(base_auc).sum()
static_attn_p = (oof_df.values * w).sum(axis=1)
fusion_results['StaticAttnWeighted'] = evaluate_pred(static_attn_p)

log("\n=== FUSION RESULTS (5-fold OOF) ===")
for k, v in fusion_results.items():
    log(f"  {k:22s} Acc={v['accuracy']:.4f} F1={v['f1']:.4f} "
        f"AUC={v['auc']:.4f} Rec={v['recall']:.4f} Prec={v['precision']:.4f}")

# ---------- 10. SHAP 关键因子 (XGBoost) ----------
log("Computing SHAP on XGBoost ...")
xgb_m = models['XGBoost']
xgb_m.fit(X, y)
expl = shap.TreeExplainer(xgb_m)
sv = expl.shap_values(X)
# 二分类取正类
if isinstance(sv, list):
    sv = sv[1]
mean_abs = np.abs(sv).mean(axis=0)
shap_rank = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs}).sort_values(
    'mean_abs_shap', ascending=False)
shap_rank.head(15).to_csv(os.path.join(OUT, "shap_top.csv"), index=False)
log("Top SHAP features:\n", shap_rank.head(10).to_string(index=False))

# ---------- 11. 消融: 静态 / +测评 / +VLE ----------
log("Ablation study ...")
def run_ablation(cols_num, tag):
    Xa = np.hstack([df[cols_num].astype(float).values,
                    pd.get_dummies(df[cat_cols].astype(str)).values])
    res = {}
    for nm, md in models.items():
        p = cross_val_predict(md, Xa, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
        res[nm] = evaluate_pred(p)
    # 融合
    pa = np.mean([res[nm]['auc'] and cross_val_predict(md, Xa, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
                  for nm, md in models.items()], axis=0) if False else None
    return res

abl = {}
# 仅静态
static_only_num = ['date_registration', 'num_of_prev_attempts', 'studied_credits']
Xs = np.hstack([df[static_only_num].astype(float).values,
                pd.get_dummies(df[cat_cols].astype(str)).values])
abl['static_only'] = {}
for nm, md in models.items():
    p = cross_val_predict(md, Xs, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
    abl['static_only'][nm] = evaluate_pred(p)

# 静态+测评
sa_num = static_only_num + ['n_assess_attempted', 'mean_score', 'first_score', 'early_score', 'mean_days_submitted']
Xsa = np.hstack([df[sa_num].astype(float).values,
                 pd.get_dummies(df[cat_cols].astype(str)).values])
abl['static_assess'] = {}
for nm, md in models.items():
    p = cross_val_predict(md, Xsa, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
    abl['static_assess'][nm] = evaluate_pred(p)

# 全部 (含VLE)
abl['full'] = {nm: evaluate_pred(oof[nm]) for nm in models}

# 融合在三种设置下的 AUC
def fusion_auc_on_setting(preds_dict):
    mx = np.column_stack([preds_dict[n] for n in models])
    # stacking LR OOF on this setting
    sk = StratifiedKFold(5, shuffle=True, random_state=42)
    o = np.zeros(len(y))
    for tr, va in sk.split(mx, y):
        mr = LogisticRegression(max_iter=2000, class_weight='balanced')
        mr.fit(mx[tr], y[tr]); o[va] = mr.predict_proba(mx[va])[:, 1]
    return roc_auc_score(y, o)

# 为静态/静态+测评 计算融合OOF AUC (简化: 用 cross_val_predict 矩阵)
def fusion_auc_cv(Xa):
    mx = np.column_stack([cross_val_predict(md, Xa, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
                          for md in models.values()])
    sk = StratifiedKFold(5, shuffle=True, random_state=42)
    o = np.zeros(len(y))
    for tr, va in sk.split(mx, y):
        mr = LogisticRegression(max_iter=2000, class_weight='balanced')
        mr.fit(mx[tr], y[tr]); o[va] = mr.predict_proba(mx[va])[:, 1]
    return roc_auc_score(y, o)

abl_summary = {
    'static_only_AUC': fusion_auc_cv(Xs),
    'static_assess_AUC': fusion_auc_cv(Xsa),
    'full_AUC': fusion_results['Stacking(LR)']['auc'],
}
log("Ablation fusion AUC:", abl_summary)

# ---------- 12. 保存汇总 ----------
summary = {
    'n_students_enrollments': int(len(df)),
    'positivity_rate': round(float(y.mean()), 4),
    'features_used_vle': USE_VLE,
    'base_oof_auc': {n: round(float(roc_auc_score(y, oof[n])), 4) for n in models},
    'fusion': {k: {m: round(float(v[m]), 4) for m in v} for k, v in fusion_results.items()},
    'ablation_fusion_auc': {k: round(float(v), 4) for k, v in abl_summary.items()},
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log("Saved summary.json. ALL DONE in", round(time.time()-t0, 1), "s")
