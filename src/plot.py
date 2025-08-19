import matplotlib.pyplot as plt
import numpy as np

def plot_ev_bids_and_price(model, dam_price, energy_bids=None, output_path=None):
    """
    简洁绘制 EV 的 DAM Energy Bids 与价格曲线
    """
    # 1. 提取 EV bids（96个时间步）  这里取所有场景的平均值  因为场景概率均等
    ev_bids = np.zeros(96)
    
    # 如果直接传入了energy_bids，优先使用传入的值

    # 否则尝试从模型中获取
    for t in range(96):
        var = model.model.getVarByName(f"P_ev0_total[{t}]")
        if var is not None:
            ev_bids[t] = var.X
 
    # 2. 提取 DAM 价格（96 点）
    prices = dam_price['price'].values[:96]

    # 3. 时间轴（0:00 到 23:45，每15分钟）
    hours = np.arange(0, 24, 0.25)

    # 4. 开始绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(hours, ev_bids, width=0.2, color='orange', alpha=0.7, label='EV bids')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('EV Bids (MW)', color='black')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))

    ax2 = ax1.twinx()
    ax2.plot(hours, prices, color='blue', linewidth=1.5, label='Energy Price')
    ax2.set_ylabel('Energy Price (€/MWh)', color='blue')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_title('EV Bids and DAM Price')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_ev_regulation_bids(model, output_path=None, case_name="Case I"):
    """
    绘制EV的调频上调(RU)和下调(RD)投标图
    
    Args:
        model: 已求解的优化模型
        output_path: 输出图像路径，如果为None则显示图像
        case_name: 案例名称，用于图表标题
    """
    # 1. 提取调频上调和下调投标数据
    reg_up = np.zeros(96)   # 上调容量
    reg_dn = np.zeros(96)   # 下调容量
    
    # 从模型中提取数据
    for t in range(96):
        # 获取上调容量变量
        ru_var = model.model.getVarByName(f"R_ev_up[{t}]")
        if ru_var is not None:
            reg_up[t] = ru_var.X
            
        # 获取下调容量变量
        rd_var = model.model.getVarByName(f"R_ev_dn[{t}]")
        if rd_var is not None:  
            reg_dn[t] = rd_var.X
    
    # 2. 创建时间轴
    hours = np.arange(0, 24, 0.25)
    
    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 4. 绘制上调和下调曲线
    # 注意：将下调容量转为负值以便在图中区分上下调
    ax.step(hours, reg_up, where='post', color='gold', linestyle='-', label=f'{case_name}(RU)')
    ax.step(hours, -reg_dn, where='post', color='gold', linestyle='--', label=f'{case_name}(RD)')
    
    # 5. 设置坐标轴和标签
    ax.set_xlabel('Time/h')
    ax.set_ylabel('Regulation Bids/MW')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    
    # 设置Y轴范围，确保上下对称
    y_max = max(max(reg_up), max(reg_dn)) * 1.2
    ax.set_ylim(-y_max, y_max)
    
    # 添加水平零线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. 添加图例
    ax.legend()
    
    # 7. 保存或显示图像
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_es_bids_and_price(model, dam_price, energy_bids=None, output_path=None):
    """
    简洁绘制 ES 的 DAM Energy Bids 与价格曲线
    """
    # 1. 提取 ES bids（96个时间步）  这里取所有场景的平均值  因为场景概率均等
    es_bids = np.zeros(96)
    
    # 如果直接传入了energy_bids，优先使用传入的值

    # 否则尝试从模型中获取
    for t in range(96):
        var = model.model.getVarByName(f"P_es0[{t}]")
        if var is not None:
            es_bids[t] = var.X
 
    # 2. 提取 DAM 价格（96 点）
    prices = dam_price['price'].values[:96]

    # 3. 时间轴（0:00 到 23:45，每15分钟）
    hours = np.arange(0, 24, 0.25)

    # 4. 开始绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(hours, es_bids, width=0.2, color='orange', alpha=0.7, label='ES bids')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('ES Bids (MW)', color='black')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))

    ax2 = ax1.twinx()
    ax2.plot(hours, prices, color='blue', linewidth=1.5, label='Energy Price')
    ax2.set_ylabel('Energy Price (€/MWh)', color='blue')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_title('ES Bids and DAM Price')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()



def plot_es_regulation_bids(model, output_path=None, case_name="Case III"):
    """
    绘制 ES 的 RU / RD 调频投标图
    """
    reg_up = np.zeros(96)
    reg_dn = np.zeros(96)

    for t in range(96):
        ru_var = model.model.getVarByName(f"R_es_up[{t}]")
        if ru_var is not None:
            reg_up[t] = ru_var.X
        rd_var = model.model.getVarByName(f"R_es_dn[{t}]")
        if rd_var is not None:
            reg_dn[t] = rd_var.X

    hours = np.arange(0, 24, 0.25)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.step(hours, reg_up, where='post', linestyle='-', color='purple', label=f"{case_name}(RU)")
    ax.step(hours, -reg_dn, where='post', linestyle='--', color='purple', label=f"{case_name}(RD)")

    ax.set_xlabel('Time/h')
    ax.set_ylabel('Regulation Bids/MW')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_ylim(-3.2, 3.2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_agc_signal(agc_scenario, *, direction: str = 'dn', output_path: str = None, title: str = None):
    """绘制 AGC 信号 l(t) 区域图。

    Args:
        agc_scenario: 单个AGC场景的 DataFrame，包含列 'agc_up' 与 'agc_dn'（MW/MW）。
        direction: 'up' or 'dn'。'dn' 将以负方向填充显示。
        output_path: 保存路径；若为 None 则直接显示。
        title: 图标题；若为 None 会根据方向自动生成。
    """

    T = len(agc_scenario)
    hours = np.arange(0, 24, 24 / T)

    if direction.lower() == 'up':
        y = agc_scenario['agc_up'].values[:T]
        y_label = 'Regulation Up AGC Signal'
        default_title = '(AGC Up)'
        lower, upper = 0.0, max(0.1, float(np.max(y))) * 1.05
        fill_from, fill_to = 0, y
    else:
        mag = agc_scenario['agc_dn'].values[:T]
        y = -mag
        y_label = 'Regulation Down AGC Signal'
        default_title = '(AGC Down)'
        lower, upper = min(-0.1, float(np.min(y))) * 1.05, 0.0
        fill_from, fill_to = y, 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hours, fill_from, fill_to, step='post', color='gray', alpha=0.7)
    ax.set_xlabel('Time/h')
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(lower, upper)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_title(title or default_title)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_agc_signal_both(agc_scenario, *, output_path: str = None, title: str = None):
    """在同一张图中同时绘制 AGC Up 与 AGC Down。

    Args:
        agc_scenario: 单个AGC场景 DataFrame，包含列 'agc_up' 与 'agc_dn'（均为非负幅值）。
        output_path: 保存路径；若为 None 则直接显示。
        title: 图标题；默认为 'AGC Up & Down'。
    """
    T = len(agc_scenario)
    hours = np.arange(0, 24, 24 / T)

    up = agc_scenario['agc_up'].values[:T]
    dn = agc_scenario['agc_dn'].values[:T]

    y_up = up  # 向上为正
    y_dn = -dn  # 向下为负

    y_max = float(max(np.max(y_up) if y_up.size else 0.0,
                      np.max(dn) if dn.size else 0.0))
    y_lim = 1.05 * (y_max if y_max > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    # 上调区域（浅灰）
    ax.fill_between(hours, 0, y_up, step='post', color='0.7', alpha=0.9, label='AGC Up')
    # 下调区域（深灰）
    ax.fill_between(hours, y_dn, 0, step='post', color='0.4', alpha=0.9, label='AGC Down')

    ax.set_xlabel('Time/h')
    ax.set_ylabel('AGC Signal')
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(-y_lim, y_lim)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.legend(loc='upper right')
    ax.set_title(title or 'AGC Up & Down')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_es_energy_change(model, dam_price=None, capacity_price=None, output_path=None, case_name="Case II"):
    """
    绘制 ES 的 energy 随时间变化图，并可选叠加 DAM 价格与容量价格（上/下）。
    - 左轴：ES1/ES2 能量（MWh）
    - 右轴：DAM 价格与容量价格（€/MWh）
    """
    T = getattr(model, 'T', 96)
    W = getattr(model, 'num_scenarios', 1)
    energy1 = np.zeros(T)
    energy2 = np.zeros(T)

    for t in range(T):
        # 按场景平均能量
        val1_sum, val2_sum = 0.0, 0.0
        found1, found2 = False, False
        for w in range(W):
            var1 = model.model.getVarByName(f"E_es1[{w},{t}]")
            var2 = model.model.getVarByName(f"E_es2[{w},{t}]")
            if var1 is not None:
                val1_sum += var1.X
                found1 = True
            if var2 is not None:
                val2_sum += var2.X
                found2 = True
        if found1:
            energy1[t] = val1_sum / W
        if found2:
            energy2[t] = val2_sum / W

    hours = np.arange(0, 24, 24 / T)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：ES 能量
    if energy1.any():
        ax1.step(hours, energy1, where='post', linestyle='-', color='tab:blue', label=f"{case_name} (ES1)")
    if energy2.any():
        ax1.step(hours, energy2, where='post', linestyle='--', color='tab:blue', label=f"{case_name} (ES2)")

    ax1.set_xlabel('Time/h')
    ax1.set_ylabel('Energy (MWh)')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_ylim(0, 3.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 右轴：价格曲线（可选）
    # ax2 = ax1.twinx()
    # any_price = False
    # if dam_price is not None and 'price' in dam_price.columns:
    #     dam = dam_price['price'].values[:T]
    #     ax2.plot(hours, dam, color='tab:green', linewidth=1.5, label='DAM Price (€/MWh)')
    #     any_price = True
    # if capacity_price is not None and {
    #     'afrr_up_cap_price', 'afrr_dn_cap_price'
    # }.issubset(capacity_price.columns):
    #     cap_up = capacity_price['afrr_up_cap_price'].values[:T]
    #     cap_dn = capacity_price['afrr_dn_cap_price'].values[:T]
    #     ax2.plot(hours, cap_up, color='tab:red', linewidth=1.5, label='Capacity Up (€/MWh)')
    #     ax2.plot(hours, cap_dn, color='tab:red', linestyle='--', linewidth=1.5, label='Capacity Down (€/MWh)')
    #     any_price = True
    # if any_price:
    #     ax2.set_ylabel('Price (€/MWh)')

    # 图例
    ax1.legend(loc='upper left')
    # if any_price:
    #     ax2.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_ev_reg_bids_and_capacity_price(model, capacity_price, output_path=None, case_name="Case"):
    """
    将 EV 的调频投标（RU/RD）与容量价格（上/下）绘制到同一张图。
    - 左轴：EV RU/RD（MW）
    - 右轴：容量价格（€/MWh）
    """
    T = getattr(model, 'T', 96)
    hours = np.arange(0, 24, 24 / T)

    # 提取 EV RU/RD（场景无关变量）
    reg_up = np.zeros(T)
    reg_dn = np.zeros(T)
    for t in range(T):
        ru_var = model.model.getVarByName(f"R_ev_up[{t}]")
        rd_var = model.model.getVarByName(f"R_ev_dn[{t}]")
        if ru_var is not None:
            reg_up[t] = ru_var.X
        if rd_var is not None:
            reg_dn[t] = rd_var.X

    # 提取容量价格
    cap_up = capacity_price['afrr_up_cap_price'].values[:T]
    cap_dn = capacity_price['afrr_dn_cap_price'].values[:T]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：EV RU/RD
    ax1.step(hours, reg_up, where='post', color='orange', label=f'{case_name} EV RU (MW)')
    ax1.step(hours, -reg_dn, where='post', color='orange', linestyle='--', label=f'{case_name} EV RD (MW)')
    ax1.set_xlabel('Time/h')
    ax1.set_ylabel('EV Regulation Bids (MW)')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))

    # 右轴：容量价格
    ax2 = ax1.twinx()
    ax2.plot(hours, cap_up, color='blue', linewidth=1.5, label='Capacity Up Price (€/MWh)')
    ax2.plot(hours, cap_dn, color='blue', linestyle='--', linewidth=1.5, label='Capacity Down Price (€/MWh)')
    ax2.set_ylabel('Capacity Price (€/MWh)', color='blue')

    # 图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_title('EV Regulation Bids and Capacity Prices')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
