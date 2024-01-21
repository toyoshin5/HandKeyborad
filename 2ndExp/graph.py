import matplotlib.pyplot as plt

# CPMとEPCのデータ
cpm_data = [20.588, 17.948, 20.000, 21.863, 19.811, 20.388, 22.018, 21.429, 20.192, 19.834]
epc_data = [0.1142857143, 0.1142857143, 0.1643835616, 0.15, 0.08571428571, 0.1428571429, 0.075, 0.15, 0.1428571429, 0.075]

# ボックスプロットの描画
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

# CPMのボックスプロット
axes[0].boxplot(cpm_data, vert=True, patch_artist=True)
axes[0].set_title('CPM (Characters Per Minute)')
axes[0].set_ylim([10, 25])

# 平均値を表示
mean_cpm = round(sum(cpm_data) / len(cpm_data), 2)
axes[0].text(0.65, mean_cpm, f'Mean: {mean_cpm}', color='black')
# 平均の記号を表示
axes[0].plot(1, mean_cpm, marker='^', color='red')


# EPCのボックスプロット
axes[1].boxplot(epc_data, vert=True, patch_artist=True)
axes[1].set_title('EPC (Error Per Character)')
axes[1].set_ylim([0, 0.2])

# 平均値を表示
mean_epc = round(sum(epc_data) / len(epc_data), 3)
axes[1].text(0.65, mean_epc, f'Mean: {mean_epc}', color='black')
# 平均の記号を表示
axes[1].plot(1, mean_epc, marker='^', color='red')


plt.tight_layout()
plt.show()
