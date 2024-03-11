y_pred_total = y_pred_total.set_index('idx')
predictions = y_pred_total.join(merge3)
predictions = predictions[['reaction','metabolite','class_prediction']]
predictions.to_excel(f"/content/gdrive/MyDrive/phd/PMI/EtaFunction/predictions_{run}_{labeling}.xlsx", index=None )
predictions

print(predictions['class_prediction'].value_counts()[0], predictions['class_prediction'].value_counts()[1])
positive_mpis = predictions.loc[predictions['class_prediction'] == 1]
positive_mpis
positive_mpis_count = pd.DataFrame(positive_mpis.groupby(['metabolite'])['class_prediction'].count())
positive_mpis_count = positive_mpis_count.sort_values(by='class_prediction',ascending=False)
positive_mpis_count.head(15)

x = positive_mpis_count[positive_mpis_count.class_prediction >= 10]
len(x)

import matplotlib

cmap = matplotlib.cm.get_cmap('Set3')

plt.figure(figsize=(40, 6))
# ax1 = plt.subplot(2,2,1)


plt1 = x.plot(kind="barh", width=0.8, edgecolor='white', color=cmap(0.8), linewidth=3,legend=0)
# ax1.set_ylabel("F1-score")
# ax1.tick_params(axis='x', labelrotation = 0)
# ax1.set_title('A', loc='left')

# handles, labels = axs[1,1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', fancybox=False, ncol=1, bbox_to_anchor=(0.96, 0.955),frameon=False)
# plt.tight_layout() # h_pad=2
# plt.ylim(0, 1)

plt.xlim(0, 200)
plt.show()