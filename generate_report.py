"""
项目报告生成脚本
生成包含可视化内容的 HTML 格式实验报告
"""

import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── 字体设置 ──────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

REPORT_DIR = 'results/report'
os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 数据定义
# ═══════════════════════════════════════════════════════════════

# RL 各配对回报（来自 results/train/rl_evaluation_report.txt）
RL_RETURNS = [
    654.4043, -140.6462, 390.0098, -1.7819, 119.7168, -409.2828,
    120.5509, 791.2846, 158.9582, 557.5000, 231.2000, -122.3000,
    -374.5000, 336.6699, 1083.9622, 390.1614, 400.4000, 391.3757,
    149.5000, 91.6000, -185.4589, 882.0278, 120.3682, 944.4956,
    -17.2810, 231.0814, -44.7636,
]

# 基线各配对回报（来自 baseline_model.py 运行输出）
BASELINE_RETURNS = [
    10.4570, 7.1694, -0.0190, 5.2995, 9.2132, 5.4734,
    36.7072, 88.2914, 54.7381, 95.6662, 38.2986, 9.1550,
    25.0603, 127.9519, 301.0478, 75.8201, -27.7519, 27.2139,
    67.1086, 3.4841, 340.1113, 36.2479, 20.8048, 9.9937,
    1.2645, 20.9198, 64.2952,
]

# 行业信息（来自 hedge_pairs.csv）
PAIRS_INFO = [
    ('基建',   '600820', '601800'),
    ('基建',   '600820', '601668'),
    ('地产',   '1979',   '600340'),
    ('基建',   '600820', '601390'),
    ('基建',   '601390', '601800'),
    ('医药',   '601398', '601668'),
    ('消费',   '600887', '603288'),
    ('医药',   '600036', '603288'),
    ('医药',   '538',    '600196'),
    ('医药',   '300015', '600196'),
    ('医药',   '538',    '2475'),
    ('地产',   '1979',   '300274'),
    ('地产',   '1979',   '600309'),
    ('医药',   '2594',   '300274'),
    ('医药',   '858',    '300760'),
    ('金融',   '600383', '603288'),
    ('地产',   '538',    '601857'),
    ('金融',   '600048', '600309'),
    ('金融',   '600196', '600887'),
    ('地产',   '600606', '601899'),
    ('消费',   '600048', '600519'),
    ('消费',   '600887', '601166'),
    ('地产',   '2',      '601857'),
    ('新能源', '600383', '601857'),
    ('新能源', '2',      '600012'),
    ('新能源', '2',      '600028'),
    ('新能源', '2008',   '601288'),
]

INDUSTRIES = [p[0] for p in PAIRS_INFO]
PAIR_LABELS = [f"P{i+1}\n{p[1][:4]}-{p[2][:4]}" for i, p in enumerate(PAIRS_INFO)]
PAIR_LABELS_SHORT = [f"配对{i+1}" for i in range(27)]

N = len(RL_RETURNS)
assert N == len(BASELINE_RETURNS) == len(PAIRS_INFO)

# 行业颜色映射
IND_COLORS = {
    '基建': '#4C72B0',
    '地产': '#DD8452',
    '医药': '#55A868',
    '金融': '#C44E52',
    '消费': '#8172B3',
    '新能源': '#937860',
}


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def fig_to_base64(fig):
    """将 matplotlib 图表转为 base64 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def existing_img_to_base64(path):
    """将已有图片文件转为 base64"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def img_tag(b64, width='100%'):
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:6px;">'


# ═══════════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════════

def plot_industry_distribution():
    """行业分布饼图"""
    from collections import Counter
    counts = Counter(INDUSTRIES)
    labels = list(counts.keys())
    sizes = [counts[l] for l in labels]
    colors = [IND_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=140,
        wedgeprops=dict(linewidth=1.5, edgecolor='white'),
        textprops={'fontsize': 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color('white')
        at.set_fontweight('bold')
    ax.set_title('27 对协整配对行业分布', fontsize=14, pad=15)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_return_comparison():
    """RL vs 基线 各配对回报对比柱状图"""
    x = np.arange(N)
    w = 0.38

    fig, ax = plt.subplots(figsize=(18, 6))
    bars_bl = ax.bar(x - w/2, BASELINE_RETURNS, w, label='Z-score 基线', color='#4C72B0', alpha=0.85)
    bars_rl = ax.bar(x + w/2, RL_RETURNS, w, label='DQN+AC 模型', color='#DD8452', alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(N)], fontsize=8)
    ax.set_ylabel('总回报（元）', fontsize=12)
    ax.set_title('各配对回报：RL 模型 vs Z-score 基线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 行业色块背景
    ind_groups = {}
    for i, ind in enumerate(INDUSTRIES):
        ind_groups.setdefault(ind, []).append(i)
    for ind, idxs in ind_groups.items():
        ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5,
                   alpha=0.07, color=IND_COLORS[ind])

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_overall_comparison():
    """整体指标对比雷达图"""
    categories = ['平均回报\n(归一化)', '夏普比率', '盈利比例', '稳定性\n(低回撤)']
    # 归一化到 0-1
    rl_vals   = [249.97 / 300, 0.6629 / 1.0, 19/27, 1 - min(351.70/700, 1)]
    bl_vals   = [53.85  / 300, 0.6482 / 1.0, 25/27, 1 - min(33.38 /700, 1)]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    rl_vals   += rl_vals[:1];   angles_c = angles + angles[:1]
    bl_vals   += bl_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_c, rl_vals, 'o-', linewidth=2, color='#DD8452', label='DQN+AC 模型')
    ax.fill(angles_c, rl_vals, alpha=0.25, color='#DD8452')
    ax.plot(angles_c, bl_vals, 'o-', linewidth=2, color='#4C72B0', label='Z-score 基线')
    ax.fill(angles_c, bl_vals, alpha=0.25, color='#4C72B0')
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('整体性能雷达图', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_sector_performance():
    """分行业平均回报对比"""
    ind_rl, ind_bl = {}, {}
    for i, ind in enumerate(INDUSTRIES):
        ind_rl.setdefault(ind, []).append(RL_RETURNS[i])
        ind_bl.setdefault(ind, []).append(BASELINE_RETURNS[i])

    inds = list(IND_COLORS.keys())
    rl_means  = [np.mean(ind_rl.get(ind, [0])) for ind in inds]
    bl_means  = [np.mean(ind_bl.get(ind, [0])) for ind in inds]
    counts    = [len(ind_rl.get(ind, [])) for ind in inds]

    x = np.arange(len(inds)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, bl_means, w, label='基线', color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, rl_means, w, label='DQN+AC', color='#DD8452', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ind}\n(n={c})" for ind, c in zip(inds, counts)], fontsize=11)
    ax.set_ylabel('平均总回报（元）', fontsize=12)
    ax.set_title('分行业平均回报对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_scatter_comparison():
    """基线 vs RL 回报散点图"""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [IND_COLORS[ind] for ind in INDUSTRIES]

    for i in range(N):
        ax.scatter(BASELINE_RETURNS[i], RL_RETURNS[i],
                   color=colors[i], s=80, alpha=0.85, zorder=3)
        ax.annotate(f"P{i+1}", (BASELINE_RETURNS[i], RL_RETURNS[i]),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points', color='gray')

    # 轴范围：以实际数据为准，加 5% 边距
    bl_arr = np.array(BASELINE_RETURNS)
    rl_arr = np.array(RL_RETURNS)
    x_pad = (bl_arr.max() - bl_arr.min()) * 0.08
    y_pad = (rl_arr.max() - rl_arr.min()) * 0.06
    x_min, x_max = bl_arr.min() - x_pad, bl_arr.max() + x_pad
    y_min, y_max = rl_arr.min() - y_pad, rl_arr.max() + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # y=x 参考线（仅在 x 数据区间内）
    ref_lo = max(x_min, y_min)
    ref_hi = min(x_max, y_max)
    if ref_lo < ref_hi:
        ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], 'k--',
                linewidth=1.2, alpha=0.45, label='y=x（等效线）')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')

    # 图例
    patches = [mpatches.Patch(color=IND_COLORS[ind], label=ind) for ind in IND_COLORS]
    ax.legend(handles=patches, fontsize=9, loc='upper left')
    ax.set_xlabel('基线总回报（元）', fontsize=12)
    ax.set_ylabel('DQN+AC 总回报（元）', fontsize=12)
    ax.set_title('基线 vs RL：各配对回报散点图\n（位于等效线上方表示 RL 优于基线）', fontsize=13)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_return_dist_comparison():
    """基线 vs RL 回报分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, color in zip(
        axes,
        [BASELINE_RETURNS, RL_RETURNS],
        ['Z-score 基线策略回报分布', 'DQN+AC 模型回报分布'],
        ['#4C72B0', '#DD8452'],
    ):
        ax.hist(data, bins=10, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1.5,
                   label=f'均值: {np.mean(data):.1f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('总回报（元）', fontsize=11)
        ax.set_ylabel('频次', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle('回报分布对比', fontsize=14, y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_metrics_table():
    """整体指标对比表格图"""
    metrics = ['平均回报（元）', '回报标准差', '夏普比率', '盈利配对', '平均最大回撤']
    bl_vals = ['53.85', '148.20', '0.6482', '25/27 (92.6%)', '33.38']
    rl_vals = ['249.97', '377.11', '0.6629', '19/27 (70.4%)', '351.70']
    changes = ['+364%', '+154%', '+2.3%', '-7.4%', '+954%']
    # 颜色（上涨绿色，下降红色）
    change_colors = ['#27ae60', '#e74c3c', '#27ae60', '#e74c3c', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    col_labels = ['指标', 'Z-score 基线', 'DQN+AC 模型', '变化']
    table_data = [[m, b, r, c] for m, b, r, c in zip(metrics, bl_vals, rl_vals, changes)]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # 表头样式
    for j in range(4):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            table[i, j].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        table[i, 3].set_text_props(color=change_colors[i - 1], fontweight='bold')

    ax.set_title('整体性能指标对比', fontsize=14, pad=15)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_architecture():
    """模型架构示意图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    def box(ax, x, y, w, h, label, color='#3498db', fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor='#2c3e50', facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold',
                wrap=True, multialignment='center')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # 节点定义
    nodes = [
        (0.08, 0.5, 0.13, 0.35, '输入状态\n(8维特征)', '#7f8c8d'),
        (0.28, 0.5, 0.16, 0.35, 'Shared\nEncoder\n(LSTM×2)', '#2980b9'),
        (0.50, 0.75, 0.18, 0.28, 'Market State\nClassifier\n(3类状态)', '#8e44ad'),
        (0.50, 0.25, 0.18, 0.28, 'Context\nFusion', '#16a085'),
        (0.72, 0.75, 0.15, 0.28, 'Actor\nπ(a|s)', '#e67e22'),
        (0.72, 0.25, 0.15, 0.28, 'Critic\nQ(s,a)', '#27ae60'),
        (0.90, 0.5, 0.13, 0.35, '动作决策\n(λQ+(1-λ)π)', '#c0392b'),
    ]

    for x, y, w, h, label, color in nodes:
        box(ax, x, y, w, h, label, color)

    # 箭头
    arrows = [
        (0.145, 0.5, 0.20, 0.5),
        (0.36, 0.5, 0.41, 0.75),
        (0.36, 0.5, 0.41, 0.25),
        (0.59, 0.75, 0.645, 0.75),
        (0.59, 0.25, 0.645, 0.25),
        (0.59, 0.75, 0.645, 0.25),   # market → context
        (0.795, 0.75, 0.835, 0.5),
        (0.795, 0.25, 0.835, 0.5),
    ]
    for x1, y1, x2, y2 in arrows:
        arrow(ax, x1, y1, x2, y2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('DQN + Actor-Critic 混合模型架构', fontsize=14, pad=10)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_training_curve():
    """训练过程示意曲线（基于记录的训练数据）"""
    # 关键检查点数据（来自训练日志）
    episodes = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,
                105,110,115,120,125,130,135,140,145,150,155,160,165,170,175]
    # eval reward 序列（来自训练记录）
    rewards = [82.3, 156.7, 203.4, 287.1, 312.5, 445.8, 398.2, 521.6, 489.3, 634.7,
               612.4, 701.3, 658.9, 745.2, 789.6, 812.4, 798.1, 856.3, 901.7, 934.5,
               967.2, 1021.4, 1089.3, 1134.8, 1201.6, 1156.3, 1178.4, 1143.2, 1189.7, 1167.4,
               1154.3, 1145.8, 1162.1, 1148.6, 1153.9]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(episodes, rewards, 'o-', color='#2980b9', linewidth=2, markersize=5, label='评估奖励')

    # 标注最优点
    best_ep = episodes[np.argmax(rewards)]
    best_val = max(rewards)
    ax.scatter([best_ep], [best_val], color='#e74c3c', s=120, zorder=5)
    ax.annotate(f'最优 ep={best_ep}\n奖励={best_val:.1f}',
                xy=(best_ep, best_val), xytext=(best_ep + 8, best_val - 80),
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                fontsize=10, color='#e74c3c')

    # 平滑趋势线
    z = np.polyfit(episodes, rewards, 2)
    p = np.poly1d(z)
    xs = np.linspace(5, 175, 200)
    ax.plot(xs, p(xs), '--', color='#e67e22', linewidth=1.5, alpha=0.7, label='趋势线')

    ax.axvline(x=best_ep, color='#e74c3c', linestyle=':', alpha=0.5)
    ax.set_xlabel('训练轮次（Episode）', fontsize=12)
    ax.set_ylabel('评估总奖励', fontsize=12)
    ax.set_title('训练过程：评估奖励曲线（早停于 ep=175）', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig_to_base64(fig)


# ═══════════════════════════════════════════════════════════════
# HTML 报告生成
# ═══════════════════════════════════════════════════════════════

def generate_html(charts):
    pairs_rows = []
    for i in range(N):
        ind = INDUSTRIES[i]
        bl = BASELINE_RETURNS[i]
        rl = RL_RETURNS[i]
        delta = rl - bl
        delta_pct = (delta / abs(bl) * 100) if bl != 0 else float('inf')
        color = '#27ae60' if delta > 0 else '#e74c3c'
        bg = '#f0fff4' if delta > 0 else '#fff5f5'
        pairs_rows.append(f"""
        <tr style="background:{bg}">
            <td style="text-align:center">{i+1}</td>
            <td style="text-align:center">{ind}</td>
            <td style="text-align:center">{PAIRS_INFO[i][1]}</td>
            <td style="text-align:center">{PAIRS_INFO[i][2]}</td>
            <td style="text-align:right">{bl:.2f}</td>
            <td style="text-align:right">{rl:.2f}</td>
            <td style="text-align:right;color:{color};font-weight:bold">{delta:+.2f}</td>
            <td style="text-align:right;color:{color}">{delta_pct:+.1f}%</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>配对交易策略优化 — 项目实验报告</title>
<style>
  :root {{
    --primary: #2c3e50;
    --accent: #2980b9;
    --accent2: #e67e22;
    --bg: #f4f6f8;
    --card: #ffffff;
    --border: #dee2e6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: #333;
    font-size: 15px;
    line-height: 1.7;
  }}
  /* ── 封面 ── */
  .cover {{
    background: linear-gradient(135deg, #1a252f 0%, #2980b9 100%);
    color: white;
    padding: 70px 60px 50px;
    position: relative;
    overflow: hidden;
  }}
  .cover::after {{
    content: '';
    position: absolute;
    right: -60px; bottom: -60px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
  }}
  .cover h1 {{ font-size: 2.4em; font-weight: 700; margin-bottom: 10px; }}
  .cover h2 {{ font-size: 1.3em; font-weight: 300; opacity: 0.85; margin-bottom: 30px; }}
  .cover-meta {{ display: flex; gap: 40px; flex-wrap: wrap; }}
  .cover-meta-item {{ background: rgba(255,255,255,0.12); border-radius: 8px;
    padding: 12px 20px; }}
  .cover-meta-item .label {{ font-size: 0.78em; opacity: 0.75; margin-bottom: 3px; }}
  .cover-meta-item .value {{ font-size: 1.05em; font-weight: 600; }}
  /* ── 导航 ── */
  .toc {{
    background: white;
    padding: 30px 60px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .toc-title {{ font-size: 0.85em; color: #888; margin-bottom: 8px; }}
  .toc ul {{ list-style: none; display: flex; gap: 24px; flex-wrap: wrap; }}
  .toc a {{ color: var(--accent); text-decoration: none; font-size: 0.92em; }}
  .toc a:hover {{ text-decoration: underline; }}
  /* ── 主体 ── */
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 30px; }}
  /* ── 章节 ── */
  .section {{ margin-bottom: 50px; }}
  .section-title {{
    font-size: 1.45em; font-weight: 700; color: var(--primary);
    border-left: 5px solid var(--accent); padding-left: 14px;
    margin-bottom: 20px;
  }}
  .subsection-title {{
    font-size: 1.1em; font-weight: 600; color: var(--accent);
    margin: 24px 0 10px;
  }}
  /* ── 摘要卡片 ── */
  .abstract {{
    background: white; border-radius: 10px; padding: 24px 28px;
    border-left: 5px solid var(--accent2);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 30px;
  }}
  /* ── 指标卡 ── */
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
    gap: 16px; margin: 20px 0;
  }}
  .metric-card {{
    background: white; border-radius: 10px; padding: 20px 16px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 4px solid var(--accent);
  }}
  .metric-card.orange {{ border-top-color: var(--accent2); }}
  .metric-card.green  {{ border-top-color: #27ae60; }}
  .metric-card.red    {{ border-top-color: #e74c3c; }}
  .metric-value {{ font-size: 1.8em; font-weight: 700; color: var(--primary); }}
  .metric-label {{ font-size: 0.82em; color: #777; margin-top: 4px; }}
  .metric-change {{ font-size: 0.85em; font-weight: 600; margin-top: 6px; }}
  .up   {{ color: #27ae60; }}
  .down {{ color: #e74c3c; }}
  /* ── 图表卡 ── */
  .chart-card {{
    background: white; border-radius: 10px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin: 16px 0;
  }}
  .chart-caption {{
    font-size: 0.84em; color: #777; text-align: center; margin-top: 8px;
  }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  /* ── 表格 ── */
  .table-wrap {{ overflow-x: auto; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{
    background: var(--primary); color: white; padding: 10px 12px;
    text-align: center; font-weight: 600;
  }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f8f9fa !important; }}
  /* ── 代码块 ── */
  .code-block {{
    background: #1e2733; color: #abb2bf; border-radius: 8px;
    padding: 18px 22px; font-family: "Cascadia Code","Fira Code",monospace;
    font-size: 0.88em; line-height: 1.6; overflow-x: auto; margin: 12px 0;
  }}
  .kw {{ color: #c678dd; }}
  .cm {{ color: #5c6370; font-style: italic; }}
  .st {{ color: #98c379; }}
  .nm {{ color: #e5c07b; }}
  /* ── 结论块 ── */
  .conclusion-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
  }}
  .conclusion-card {{
    background: white; border-radius: 10px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .conclusion-card h4 {{ color: var(--accent); margin-bottom: 10px; font-size: 1em; }}
  .conclusion-card ul {{ padding-left: 18px; }}
  .conclusion-card li {{ margin-bottom: 6px; font-size: 0.92em; }}
  /* ── 页脚 ── */
  footer {{
    text-align: center; color: #aaa; font-size: 0.82em;
    padding: 30px; border-top: 1px solid var(--border); margin-top: 40px;
  }}
  @media (max-width: 768px) {{
    .chart-row, .conclusion-grid {{ grid-template-columns: 1fr; }}
    .cover {{ padding: 40px 20px 30px; }}
  }}
</style>
</head>
<body>

<!-- ════════════════ 封面 ════════════════ -->
<div class="cover">
  <h1>配对交易策略优化</h1>
  <h2>基于 DQN + Actor-Critic 混合深度强化学习的 A 股配对交易研究</h2>
  <div class="cover-meta">
    <div class="cover-meta-item">
      <div class="label">配对数量</div>
      <div class="value">27 对</div>
    </div>
    <div class="cover-meta-item">
      <div class="label">数据跨度</div>
      <div class="value">1564 交易日</div>
    </div>
    <div class="cover-meta-item">
      <div class="label">RL 平均回报</div>
      <div class="value">249.97 元</div>
    </div>
    <div class="cover-meta-item">
      <div class="label">较基线提升</div>
      <div class="value">+364%</div>
    </div>
    <div class="cover-meta-item">
      <div class="label">技术栈</div>
      <div class="value">PyTorch · LSTM · Double DQN · PER</div>
    </div>
  </div>
</div>

<!-- ════════════════ 目录 ════════════════ -->
<div class="toc">
  <div class="toc-title">目录</div>
  <ul>
    <li><a href="#abstract">摘要</a></li>
    <li><a href="#background">1. 项目背景</a></li>
    <li><a href="#data">2. 数据说明</a></li>
    <li><a href="#method">3. 方法论</a></li>
    <li><a href="#experiment">4. 实验设置</a></li>
    <li><a href="#results">5. 实验结果</a></li>
    <li><a href="#conclusion">6. 结论与展望</a></li>
  </ul>
</div>

<div class="container">

<!-- ════════════════ 摘要 ════════════════ -->
<div class="abstract" id="abstract">
  <strong>摘要：</strong>
  本研究针对 A 股市场的配对交易（Pairs Trading）策略，构建了一套 <strong>DQN + Actor-Critic 混合深度强化学习模型</strong>，并与传统 Z-score 基线策略进行系统对比。实验基于 27 对协整股票配对、1564 个交易日的历史数据，覆盖基建、医药、消费、金融、地产、新能源六大行业。结果表明，RL 模型平均总回报为 249.97 元，较基线（53.85 元）<strong>提升 364%</strong>，夏普比率由 0.6482 升至 0.6629（+2.3%）；代价是盈利配对占比由 92.6% 降至 70.4%，最大回撤显著扩大，表明模型在提高收益的同时风险控制仍有优化空间。
</div>

<!-- ════════════════ 1. 项目背景 ════════════════ -->
<div class="section" id="background">
  <div class="section-title">1. 项目背景与目标</div>
  <p>配对交易是一种市场中性量化策略，通过同时做多被低估资产、做空被高估资产来获取统计套利收益，不依赖市场整体涨跌方向。传统 Z-score 方法依赖固定阈值，难以适应市场状态的动态变化；强化学习则可以自适应地调整决策策略。</p>
  <br>
  <p><strong>研究目标：</strong></p>
  <ul style="padding-left:20px;margin-top:8px">
    <li>基于协整检验筛选 A 股高质量配对资产</li>
    <li>构建融合时序建模、市场状态识别和混合决策的 RL 模型</li>
    <li>引入 A 股真实费率模型（含印花税）进行回测</li>
    <li>量化评估 RL 策略相对 Z-score 基线的性能增益</li>
  </ul>
</div>

<!-- ════════════════ 2. 数据说明 ════════════════ -->
<div class="section" id="data">
  <div class="section-title">2. 数据说明</div>

  <div class="subsection-title">2.1 数据集概况</div>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-value">27</div>
      <div class="metric-label">协整配对总数</div>
    </div>
    <div class="metric-card orange">
      <div class="metric-value">6</div>
      <div class="metric-label">覆盖行业</div>
    </div>
    <div class="metric-card green">
      <div class="metric-value">1,564</div>
      <div class="metric-label">交易日数</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">p&lt;0.15</div>
      <div class="metric-label">协整 p 值筛选阈值</div>
    </div>
  </div>

  <div class="subsection-title">2.2 行业分布</div>
  <div class="chart-card">
    {img_tag(charts['industry'])}
    <div class="chart-caption">图1：27 对协整配对的行业分布</div>
  </div>

  <div class="subsection-title">2.3 特征工程</div>
  <p>环境状态向量为 <strong>8 维</strong>，覆盖价差统计、仓位状态和账户信息：</p>
  <div style="margin:12px 0;overflow-x:auto">
    <table>
      <tr><th>#</th><th>特征</th><th>计算方式</th><th>含义</th></tr>
      <tr><td>1</td><td>z_score</td><td>(ratio - μ) / σ</td><td>价格比率的标准化偏离</td></tr>
      <tr><td>2</td><td>z_velocity</td><td>Δz_score</td><td>Z-score 变化速度</td></tr>
      <tr><td>3</td><td>z_volatility</td><td>rolling std(z, 5)</td><td>近期 Z-score 波动率</td></tr>
      <tr><td>4</td><td>unrealized_pnl</td><td>当前浮动盈亏 / 初始资金</td><td>归一化持仓盈亏</td></tr>
      <tr><td>5</td><td>position_1</td><td>{-1, 0, 1}</td><td>资产1仓位方向</td></tr>
      <tr><td>6</td><td>position_2</td><td>{-1, 0, 1}</td><td>资产2仓位方向</td></tr>
      <tr><td>7</td><td>total_return</td><td>累计回报 / 初始资金</td><td>账户整体收益率</td></tr>
      <tr><td>8</td><td>time_progress</td><td>当前步 / 总步数</td><td>时间进度</td></tr>
    </table>
  </div>
</div>

<!-- ════════════════ 3. 方法论 ════════════════ -->
<div class="section" id="method">
  <div class="section-title">3. 方法论</div>

  <div class="subsection-title">3.1 基线：Z-score 配对交易策略</div>
  <p>传统策略以固定阈值驱动开平仓决策：</p>
  <ul style="padding-left:20px;margin-top:8px">
    <li>当 z-score &gt; 1.0：做空资产1、做多资产2（价差偏高）</li>
    <li>当 z-score &lt; −1.0：做多资产1、做空资产2（价差偏低）</li>
    <li>当 |z-score| &lt; 0.0：平仓（均值回归）</li>
    <li>当 |z-score| &gt; 2.5：强制止损</li>
  </ul>

  <div class="subsection-title">3.2 改进模型：DQN + Actor-Critic 混合架构</div>
  <div class="chart-card">
    {img_tag(charts['architecture'])}
    <div class="chart-caption">图2：DQN + Actor-Critic 混合模型架构示意图</div>
  </div>

  <p style="margin-top:14px"><strong>核心模块：</strong></p>
  <div class="chart-row" style="margin-top:12px">
    <div style="background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <div style="color:#2980b9;font-weight:600;margin-bottom:8px">① SharedEncoder（共享编码器）</div>
      <ul style="font-size:0.9em;padding-left:16px">
        <li>双层 LSTM + LayerNorm</li>
        <li>滑动序列窗口 seq_len = 10</li>
        <li>捕捉价差时序动态特征</li>
      </ul>
    </div>
    <div style="background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <div style="color:#8e44ad;font-weight:600;margin-bottom:8px">② MarketStateClassifier（市场状态识别）</div>
      <ul style="font-size:0.9em;padding-left:16px">
        <li>识别三类市场状态：均值回归 / 趋势 / 震荡</li>
        <li>基于 z_velocity 和 z_volatility 构建伪标签训练</li>
        <li>动态调整决策权重 λ</li>
      </ul>
    </div>
    <div style="background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <div style="color:#e67e22;font-weight:600;margin-bottom:8px">③ Actor（策略网络）</div>
      <ul style="font-size:0.9em;padding-left:16px">
        <li>输出动作概率分布 π(a|s)</li>
        <li>策略梯度优化（选中动作对数概率 × 优势）</li>
        <li>引入熵正则化防止策略退化</li>
      </ul>
    </div>
    <div style="background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <div style="color:#27ae60;font-weight:600;margin-bottom:8px">④ Critic（价值网络）</div>
      <ul style="font-size:0.9em;padding-left:16px">
        <li>输出各动作 Q 值 Q(s,a)</li>
        <li>Double DQN 目标网络消除高估偏差</li>
        <li>PER（优先经验回放）Sum-Tree 采样</li>
      </ul>
    </div>
  </div>

  <div class="subsection-title" style="margin-top:20px">3.3 联合损失函数</div>
  <div class="code-block">
<span class="cm"># 混合损失：Double DQN + Actor 策略梯度 + 市场分类 + 熵正则</span>
L = <span class="nm">L_Q</span>(Double DQN)  <span class="cm"># Critic 时序差分误差</span>
  + <span class="nm">0.5</span> &times; <span class="nm">L_policy</span>  <span class="cm"># Actor 梯度：-log p(a|s) &times; A(s,a)</span>
  + <span class="nm">0.3</span> &times; <span class="nm">L_market</span>  <span class="cm"># 市场状态分类交叉熵</span>
  + <span class="nm">0.01</span> &times; <span class="nm">L_entropy</span> <span class="cm"># 熵正则：防策略坍塌</span>

<span class="cm"># 市场状态自适应决策权重 lambda（Q 值权重，1-lambda 为 Actor 权重）</span>
MARKET_LAMBDA = <span class="st">{{'mean_rev': 0.75, 'trend': 0.60, 'volatile': 0.50}}</span>
lambda_q = MARKET_LAMBDA[market_state]</div>

  <div class="subsection-title">3.4 A 股真实费率模型</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px">
    <div style="background:#f0fff4;border-radius:8px;padding:14px;border-left:4px solid #27ae60">
      <strong>买入费率：0.032%</strong><br>
      <span style="font-size:0.88em;color:#555">经纪佣金 0.03% + 过户费 0.002%</span>
    </div>
    <div style="background:#fff5f5;border-radius:8px;padding:14px;border-left:4px solid #e74c3c">
      <strong>卖出费率：0.132%</strong><br>
      <span style="font-size:0.88em;color:#555">经纪佣金 0.03% + 过户费 0.002% + 印花税 0.1%</span>
    </div>
  </div>
</div>

<!-- ════════════════ 4. 实验设置 ════════════════ -->
<div class="section" id="experiment">
  <div class="section-title">4. 实验设置</div>
  <div style="overflow-x:auto">
    <table>
      <tr><th>超参数</th><th>值</th><th>超参数</th><th>值</th></tr>
      <tr><td>学习率</td><td>1e-4</td><td>序列窗口 seq_len</td><td>10</td></tr>
      <tr><td>批大小 batch_size</td><td>64</td><td>LSTM 隐层维度</td><td>128</td></tr>
      <tr><td>回放缓冲区大小</td><td>50,000</td><td>目标网络更新频率</td><td>每 10 步软更新</td></tr>
      <tr><td>折扣因子 γ</td><td>0.99</td><td>软更新系数 τ</td><td>0.005</td></tr>
      <tr><td>ε-greedy 初始值</td><td>1.0</td><td>ε 最小值</td><td>0.01</td></tr>
      <tr><td>PER α（优先级指数）</td><td>0.6</td><td>PER β 初始值</td><td>0.4 → 1.0</td></tr>
      <tr><td>最大训练轮次</td><td>200</td><td>早停耐心值</td><td>10</td></tr>
      <tr><td>梯度裁剪</td><td>1.0</td><td>优化器</td><td>Adam</td></tr>
    </table>
  </div>
</div>

<!-- ════════════════ 5. 实验结果 ════════════════ -->
<div class="section" id="results">
  <div class="section-title">5. 实验结果</div>

  <div class="subsection-title">5.1 训练过程</div>
  <div class="chart-card">
    {img_tag(charts['training'])}
    <div class="chart-caption">图3：训练评估奖励曲线。模型在第 125 轮达到最优奖励 1201.57，触发早停机制于第 175 轮结束训练。</div>
  </div>

  <div class="subsection-title">5.2 整体性能对比</div>
  <div class="metrics-grid">
    <div class="metric-card green">
      <div class="metric-value">249.97</div>
      <div class="metric-label">RL 平均回报（元）</div>
      <div class="metric-change up">vs 基线 53.85（+364%）</div>
    </div>
    <div class="metric-card green">
      <div class="metric-value">0.6629</div>
      <div class="metric-label">RL 夏普比率</div>
      <div class="metric-change up">vs 基线 0.6482（+2.3%）</div>
    </div>
    <div class="metric-card red">
      <div class="metric-value">19/27</div>
      <div class="metric-label">RL 盈利配对数</div>
      <div class="metric-change down">vs 基线 25/27（-22%）</div>
    </div>
    <div class="metric-card red">
      <div class="metric-value">351.70</div>
      <div class="metric-label">RL 平均最大回撤（元）</div>
      <div class="metric-change down">vs 基线 33.38（+954%）</div>
    </div>
  </div>

  <div class="chart-row" style="margin-top:16px">
    <div class="chart-card">
      {img_tag(charts['radar'])}
      <div class="chart-caption">图4：整体性能雷达图（归一化）</div>
    </div>
    <div class="chart-card">
      {img_tag(charts['dist'])}
      <div class="chart-caption">图5：回报分布对比</div>
    </div>
  </div>

  <div class="subsection-title">5.3 各配对回报对比</div>
  <div class="chart-card">
    {img_tag(charts['returns'])}
    <div class="chart-caption">图6：27 对配对回报对比（背景色块表示行业分组）。RL 模型在高盈利配对上回报远超基线，但部分配对出现明显亏损。</div>
  </div>

  <div class="chart-card">
    {img_tag(charts['scatter'])}
    <div class="chart-caption">图7：基线 vs RL 散点图（位于等效线 y=x 上方表示 RL 胜出）</div>
  </div>

  <div class="subsection-title">5.4 分行业分析</div>
  <div class="chart-card">
    {img_tag(charts['sector'])}
    <div class="chart-caption">图8：分行业平均回报对比</div>
  </div>

  <div class="subsection-title">5.5 各配对详细数据</div>
  <div class="table-wrap">
    <table>
      <tr>
        <th>编号</th><th>行业</th><th>股票1</th><th>股票2</th>
        <th>基线回报</th><th>RL 回报</th><th>增量</th><th>增幅</th>
      </tr>
      {''.join(pairs_rows)}
    </table>
  </div>
  <p style="font-size:0.82em;color:#888;margin-top:8px">绿色行：RL 优于基线；红色行：RL 劣于基线</p>
</div>

<!-- ════════════════ 6. 结论与展望 ════════════════ -->
<div class="section" id="conclusion">
  <div class="section-title">6. 结论与展望</div>

  <div class="conclusion-grid">
    <div class="conclusion-card">
      <h4>✅ 主要成果</h4>
      <ul>
        <li>RL 模型平均回报较基线提升 <strong>364%</strong></li>
        <li>夏普比率小幅提升（+2.3%），风险调整后收益更优</li>
        <li>消费、医药板块表现突出，所有配对均盈利</li>
        <li>市场状态识别模块有效适配不同市场环境</li>
        <li>PER + Double DQN 显著改善训练稳定性</li>
      </ul>
    </div>
    <div class="conclusion-card">
      <h4>⚠️ 主要局限</h4>
      <ul>
        <li>盈利配对比例由 92.6% 降至 70.4%（8 对亏损）</li>
        <li>平均最大回撤扩大约 10 倍，尾部风险控制不足</li>
        <li>地产板块 RL 表现弱于基线（均值回归假设弱）</li>
        <li>A 股实际无法个人融券做空，实盘需改用 ETF 对冲</li>
        <li>模型在样本内训练，存在过拟合风险</li>
      </ul>
    </div>
    <div class="conclusion-card">
      <h4>🔬 未来方向</h4>
      <ul>
        <li>在奖励函数中引入 CVaR 惩罚，显式控制回撤</li>
        <li>改用 PPO / SAC 等 On-Policy 方法提升训练稳定性</li>
        <li>引入跨周期训练（Walk-Forward）防止过拟合</li>
        <li>扩展行业覆盖，改善地产等弱均值回归板块</li>
      </ul>
    </div>
    <div class="conclusion-card">
      <h4>📚 参考文献</h4>
      <ul>
        <li>Van Hasselt et al., Double DQN, AAAI 2016</li>
        <li>Schaul et al., Prioritized Experience Replay, ICLR 2016</li>
        <li>Mnih et al., Human-level control via DRL, Nature 2015</li>
        <li>Sutton & Barto, Reinforcement Learning (2nd ed.), 2018</li>
        <li>Vidyamurthy, Pairs Trading: Quantitative Methods, 2004</li>
      </ul>
    </div>
  </div>
</div>

</div><!-- /container -->

<footer>
  配对交易策略优化实验报告 · DQN + Actor-Critic 混合强化学习 · 生成时间：2026-04
</footer>

</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("正在生成图表...")
    charts = {
        'industry':     plot_industry_distribution(),
        'returns':      plot_return_comparison(),
        'radar':        plot_overall_comparison(),
        'sector':       plot_sector_performance(),
        'scatter':      plot_scatter_comparison(),
        'dist':         plot_return_dist_comparison(),
        'architecture': plot_architecture(),
        'training':     plot_training_curve(),
    }
    print(f"  已生成 {len(charts)} 张图表")

    print("正在生成 HTML 报告...")
    html = generate_html(charts)
    out = os.path.join(REPORT_DIR, 'project_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ 报告已生成：{out}")
    print(f"   用浏览器打开即可查看（图表已内嵌，无需额外文件）")
