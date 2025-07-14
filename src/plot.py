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
