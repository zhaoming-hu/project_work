import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================================
#  AGC 数据（保持你给的）
# ================================
agc = np.array([
    # 10:00–10:10 上调频（正值）
    0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.045, 0.04, 0.035, 0.03,    
    -0.02, -0.025, -0.03, -0.035, -0.04, -0.045, -0.05, -0.055, 
    -0.06, -0.055, -0.05, -0.045, -0.04, -0.035, -0.03, -0.025, -0.02, -0.018,

    # 10:29–10:35 下调频（负值，橙色区）
    -0.015, -0.012, -0.01, -0.02, -0.025, -0.03, 0.00,

    # 10:35–10:45 上调频（正值）
    0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.045, 0.04,

    # 10:45–11:00 下调频（负值）
    -0.01, -0.015, -0.02, -0.025, -0.03, -0.035, -0.04, -0.038, -0.036,
    -0.034, -0.032, -0.03, -0.028, -0.026, -0.024
])


# ================================
#  SOC（无预留）
# ================================
soc_no_res = np.array([
    89.5, 89.2, 88.9, 88.6, 88.3, 88.0, 87.7, 87.4, 87.1, 86.8,    # 放电 10 min
    87.4, 88.0, 88.8, 89.6, 90.4, 91.2, 92.0, 92.8, 93.6, 94.4,
    95.2, 96.0, 96.8, 97.6, 98.4, 99.2, 100.0, 101.0, 102.0, 103.0,
    104.0, 104.5, 105.0, 105.5, 106.0,   # 最后五分钟超限
    104.0, 102.0, 100.5, 98.0, 95.5, 92.5, 90.0, 89.5, 89.5, 90.0,  # 放电
    90.8, 91.6, 92.4, 93.2, 94.0, 95.0, 96.0, 96.8, 97.5, 97.5
])


# ================================
#  SOC（有预留）
# ================================
soc_with_res = np.array([
    89.5, 89.2, 88.9, 88.6, 88.3, 88.0, 87.7, 87.4, 87.1, 86.8,
    87.4, 88.0, 88.8, 89.6, 90.4, 91.2, 92.0, 92.8, 93.6, 94.4,
    95.2, 96.0, 96.8, 97.6, 98.4, 99.2, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0,   # 容量预留，不超限
    98.5, 96.5, 94.8, 92.8, 91.0, 89.5, 88.5, 88.5, 88.8, 89.2,
    89.9, 90.8, 91.8, 92.7, 93.3, 93.8, 94.0, 94.2, 94.3, 94.5
])




# ---------- 1. 插值函数 ----------
def smooth(data, factor=5):
    # 确保有足够的点进行三次插值
    if len(data) < 4:
        kind = 'linear' # 对非常短的数据使用线性插值
    else:
        kind = 'cubic'
    t = np.arange(len(data))
    f = interp1d(t, data, kind=kind)
    t_new = np.linspace(0, len(data) - 1, len(data) * factor)
    return t_new, f(t_new)


# ---------- 2. 时间刻度（论文格式） ----------
def time_ticks(t_raw_len, factor=5):
    # 新的时间轴 t_new 的范围是 [0, t_raw_len - 1]
    # 我们希望刻度分布在这个范围 [0, 59] 内
    labels = ["10:00:00", "10:15:00", "10:30:00", "10:45:00", "11:00:00"]
    max_t_new = t_raw_len - 1
    positions = np.linspace(0, max_t_new, len(labels))
    return positions, labels


# ---------- 3. 绘制 AGC Signals ----------
def plot_agc_paper_style(agc, shade_start_min=26, shade_end_min=34, factor=5):
    t_raw_len = len(agc)
    t_new, agc_s = smooth(agc, factor=factor)
    
    # 阴影位置基于原始索引（即插值前的分钟数）
    s = shade_start_min 
    e = shade_end_min
    
    fig, ax = plt.subplots(figsize=(10, 3))

    # ----- 灰色 AGC 区域 -----
    # 对平滑后的数据使用 fill_between
    ax.fill_between(t_new, 0, agc_s, where=agc_s >= 0, facecolor='gray', alpha=0.75, interpolate=True)
    ax.fill_between(t_new, 0, agc_s, where=agc_s <= 0, facecolor='gray', alpha=0.75, interpolate=True)

    # ----- 底线 -----
    ax.axhline(0, color='black', linewidth=1)

    # ----- 橙色 shading 区域（论文风格） -----
    # t_new 的范围是 0 到 59.
    ax.axvspan(s, e, color='orange', alpha=0.4)

    # ----- 时间刻度 -----
    pos, lab = time_ticks(t_raw_len, factor)
    ax.set_xticks(pos)
    ax.set_xlabel("Time/h", fontsize=12)
    ax.set_xticklabels(lab, fontsize=10)

    # ----- y 轴范围（论文式） -----
    ax.set_ylim(-0.08, 0.08) # 调整 ylim 以看到峰值
    ax.set_ylabel("AGC Signal", fontsize=12)

    # plt.title("(b)", fontsize=14, y=-0.3) # 调整 y
    plt.tight_layout()
    plt.savefig("agc_signal_smooth.png")
    plt.close() # 关闭图像，防止在环境中重叠显示


# ---------- 4. 绘制 SOC Comparison ----------
def plot_soc_paper_style(soc_no, soc_res, shade_start_min=26, shade_end_min=34, factor=5):
    t_raw_len = len(soc_no)

    # 插值
    t1, s_no = smooth(soc_no, factor=factor)
    t2, s_res = smooth(soc_res, factor=factor)
    
    # 阴影位置
    s = shade_start_min
    e = shade_end_min

    fig, ax = plt.subplots(figsize=(10, 3))

    # ----- 两条 SOC 曲线 -----
    ax.plot(t1, s_no, color="#d4a018", linewidth=1.8, label="Without capacity reservation")
    ax.plot(t2, s_res, color="#6a9e8c", linewidth=1.8, linestyle='--', label="With capacity reservation")

    # ----- 上限线（论文式蓝线） -----
    ax.axhline(100, color='blue', linewidth=1, linestyle=':')

    # ----- 橙色 shading 区域 -----
    ax.axvspan(s, e, color='orange', alpha=0.35)

    # ----- 时间刻度 -----
    pos, lab = time_ticks(t_raw_len, factor)
    ax.set_xticks(pos)
    ax.set_xlabel("Time/h", fontsize=12)
    ax.set_xticklabels(lab, fontsize=10)

    # ----- y 轴范围 -----
    ax.set_ylim(85, 110) # 调整 ylim 以看到溢出
    ax.set_ylabel("SOC/%", fontsize=12)

    plt.legend(fontsize=10, loc="upper right")

    # plt.title("(a)", fontsize=14, y=-0.3) # 调整 y
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("soc_comparison_smooth.png")
    plt.close() # 关闭图像


# ================================
# 调用绘图函数
# ================================
# 设置全局字体大小以便更好地阅读
plt.rcParams.update({'font.size': 11})

plot_agc_paper_style(agc)
plot_soc_paper_style(soc_no_res, soc_with_res)

# 打印确认信息，告知用户文件已保存
print("Charts saved as agc_signal_smooth.png and soc_comparison_smooth.png")