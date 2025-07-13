import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_ev_bids_and_price(model, dam_price, output_path=None):
    """
    绘制EV的DAM Energy Bids和能源价格曲线
    
    Args:
        model: 已求解的优化模型
        dam_price: 日前市场价格数据 DataFrame
        output_path: 输出图像路径，如果为None则显示图像
    """
    # 提取EV bids数据 (取平均值)
    ev_bids = np.zeros(24)  # 按小时聚合
    
    # 对所有场景求平均
    if hasattr(model, 'model') and hasattr(model, 'num_scenarios'):
        for w in range(model.num_scenarios):
            for t in range(96):  # 假设模型时间间隔为15分钟，96个时间点
                h = t // 4  # 转换为小时
                if h >= 24:  # 安全检查
                    continue
                var_name = f"P_ev0_total[{w},{t}]"
                try:
                    var = model.model.getVarByName(var_name)
                    if var is not None:
                        ev_bids[h] += var.X / model.num_scenarios / 4  # 除以4是因为每小时有4个15分钟
                except Exception as e:
                    pass  # 忽略不存在的变量
    
    # 提取能源价格数据
    prices = np.zeros(24)
    # 确保dam_price有price列
    if 'price' in dam_price.columns:
        # 如果有24个价格点或者整数倍的24，直接取平均
        if len(dam_price) == 24:
            prices = dam_price['price'].values
        elif len(dam_price) % 24 == 0:
            step = len(dam_price) // 24
            for h in range(24):
                prices[h] = dam_price['price'].iloc[h*step:(h+1)*step].mean()
        else:
            # 默认情况，简单填充为常数值
            prices = np.ones(24) * dam_price['price'].mean()
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 横坐标范围
    hours = np.arange(24)
    
    # 左Y轴 - EV bids
    color = 'tab:orange'
    ax1.set_xlabel('时间/h')
    ax1.set_ylabel('DAM Energy Bids of EVs/MW', color='black')
    ax1.bar(hours, ev_bids, color=color, alpha=0.7, width=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(-1, 24)
    
    # 右Y轴 - 能源价格
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Energy price/(€/MWh)', color=color)
    ax2.plot(hours, prices, color=color, marker='s', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例和标题
    ax1.text(0.05, 0.95, 'Case 1', transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top')
    
    # 添加Energy price图例
    ax2.plot([], [], color=color, marker='s', linewidth=2, label='Energy price')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def main():
    """
    独立调用绘图函数的示例
    """
    print("请通过main.py运行完整流程。")


if __name__ == "__main__":
    main()
