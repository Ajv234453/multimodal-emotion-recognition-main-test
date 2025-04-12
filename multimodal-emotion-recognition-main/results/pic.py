import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path_2d = '2d.csv'  # 替换为2d.csv的实际路径
file_path_3d = '3d.csv'  # 替换为3d.csv的实际路径
data_2d = pd.read_csv(file_path_2d)
data_3d = pd.read_csv(file_path_3d)

# 确定两个数据集中loss和prec1的最大和最小值，以设置相同的坐标轴刻度
loss_min = min(data_2d['loss'].min(), data_3d['loss'].min())
loss_max = max(data_2d['loss'].max(), data_3d['loss'].max())
prec1_min = min(data_2d['prec1'].min(), data_3d['prec1'].min())
prec1_max = max(data_2d['prec1'].max(), data_3d['prec1'].max())


# 定义一个绘制数据的函数
def plot_data(data, title, loss_min, loss_max, prec1_min, prec1_max):
    # 基于每5个批次创建组
    data['group'] = data.index // 10
    grouped_data = data.groupby('group').agg({'loss': 'mean', 'prec1': 'mean'}).reset_index()

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制loss曲线
    color = 'tab:red'
    ax1.set_xlabel('Group (each representing 5 batches)')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(grouped_data['group'], grouped_data['loss'], color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([loss_min - (loss_max - loss_min) * 0.1, loss_max + (loss_max - loss_min) * 0.1])

    # 为prec1创建一个共享x轴的第二个y轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(grouped_data['group'], grouped_data['prec1'], color=color, linestyle='--', marker='x', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([prec1_min - (prec1_max - prec1_min) * 0.1, prec1_max + (prec1_max - prec1_min) * 0.1])

    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 标题和布局调整
    plt.title(title)
    plt.tight_layout()

    # 返回图表对象
    return fig


# 生成图表
fig1 = plot_data(data_2d, 'Training Loss and Accuracy (2D Data)', loss_min, loss_max, prec1_min, prec1_max)
fig2 = plot_data(data_3d, 'Training Loss and Accuracy (3D Data)', loss_min, loss_max, prec1_min, prec1_max)

# 将图表保存为图片
fig1.savefig('plot_2d.png', dpi=300)  # 替换为保存图片的实际路径
fig2.savefig('plot_3d.png', dpi=300)

# 显示图表，如果在本地环境运行此脚本，则可以省略保存图片的步骤
plt.show()
