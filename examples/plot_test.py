import numpy as np
import matplotlib.pyplot as plt

# ==================== 用户需替换为真实数据 ====================
# 基因数量（横坐标）
n_genes = [700, 1400, 2100, 2800, 3150]

# 运行时间（秒），3次重复，每行对应一个基因数量，每列是一次实验
run_time_data = np.array([
    [766, 858, 799],    # 700
    [3934, 4414, 4054],    # 1400
    [9465, 9532, 9799],    # 2100
    [18723, 18222, 18329], # 2800
    [23765, 24208, 22951]  # 3150
])

# 峰值内存（MB），3次重复
peak_mem_data = np.array([
    [190, 190, 190],    # 700
    [978, 741, 741],    # 1400
    [1868, 1635, 1635],    # 2100
    [2861, 2972, 2972],    # 2800
    [3485, 3745, 3745]     # 3150
])
# ============================================================

# ===== 1. 计算均值和标准差 =====
run_time_mean = np.mean(run_time_data, axis=1)
run_time_std = np.std(run_time_data, axis=1, ddof=1)

peak_mem_mean = np.mean(peak_mem_data, axis=1)
peak_mem_std = np.std(peak_mem_data, axis=1, ddof=1)

# ===== 2. log10(runtime) =====
log_run_time_data = np.log10(run_time_data)           # 每次重复先取 log10
log_time_mean = np.mean(log_run_time_data, axis=1)    # log 空间下求均值
log_time_std = np.std(log_run_time_data, axis=1, ddof=1)

# ===== 3. 创建图形 =====
fig, ax1 = plt.subplots(figsize=(8, 6))

# ===== 4. 左轴：runtime =====
# 先画每次重复的散点
for i, x in enumerate(n_genes):
    ax1.scatter(
        [x] * run_time_data.shape[1],
        log_run_time_data[i, :],
        color="tab:blue",
        alpha=0.45,
        s=40,
        zorder=3
    )

# 再画 mean ± SD
ax1.errorbar(
    n_genes,
    log_time_mean,
    yerr=log_time_std,
    fmt='o-',
    capsize=5,
    capthick=1,
    elinewidth=1,
    color='tab:blue',
    markersize=8,
    label='Runtime'
)

ax1.set_xlabel('Number of genes', fontsize=12)
ax1.set_ylabel('log10 runtime (s)', fontsize=12, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(n_genes)
ax1.set_xticklabels([str(x) for x in n_genes])
ax1.grid(True, linestyle='--', alpha=0.5)

# ===== 5. 右轴：peak memory =====
ax2 = ax1.twinx()

# 先画每次重复的散点
for i, x in enumerate(n_genes):
    ax2.scatter(
        [x] * peak_mem_data.shape[1],
        peak_mem_data[i, :],
        color="tab:red",
        alpha=0.45,
        s=40,
        marker='s',
        zorder=3
    )

# 再画 mean ± SD
ax2.errorbar(
    n_genes,
    peak_mem_mean,
    yerr=peak_mem_std,
    fmt='s--',
    capsize=5,
    capthick=1,
    elinewidth=1,
    color='tab:red',
    markersize=8,
    label='Peak memory'
)

ax2.set_ylabel('Peak memory (MB)', fontsize=12, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# ===== 6. 合并图例 =====
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

# ===== 7. 标题 =====
plt.title('Runtime and memory scaling with gene number', fontsize=14)

# ===== 8. 保存与显示 =====
plt.tight_layout()
plt.savefig('runtime_memory_scaling_scatter_sd.png', dpi=300, bbox_inches='tight')
plt.show()