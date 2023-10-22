ds = ['consep', 'monusac', 'pannuke']
ms = ['dist', 'seg_unet', 'mask_rcnn', 'hovernet']

detail = []
for dataset_name in ds:
    for model in ms:
        file_path = f'/root/autodl-tmp/archive/core/metrics/evaluate_result/{dataset_name}_{model}_detail.csv'
        data = pd.read_csv(file_path)
        data['dataset'] = dataset_name
        data['model'] = model
        detail.append(data)
overall = pd.concat(detail)[['map_50', 'map_75', 'acc', 'f1', 'dice', 'aji',
       'aji_plus', 'dq', 'sq', 'pq', 'img_names', 'dataset', 'model']]

desc =[]
for dataset in ds:
    print(dataset)
    desc_path = f"/root/autodl-tmp/datasets/{dataset}/dist_area_cnt.csv"
    data = pd.read_csv(desc_path)
    desc.append(data)
    
descs = pd.concat(desc)[['file_name', 'dist', 'area', 'cnt']]

te = pd.merge(overall, descs, left_on='img_names', right_on='file_name')


# 分类图
fig = plt.figure(dpi=400)
ax = sns.violinplot(data=overall.query("model != 'dist'"), x="model", y="acc", hue='dataset',
                inner="quart", linewidth=1
#                palette={"Yes": "b", "No": ".85"})
              )
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/desc_cls/acc.png"
ax.get_figure().savefig(fig_path, dpi = 400)

fig = plt.figure(dpi=400)
ax = sns.violinplot(data=overall.query("model != 'dist'"), x="model", y="f1", hue='dataset',
                inner="quart", linewidth=1
#                palette={"Yes": "b", "No": ".85"})
              )
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/desc_cls/f1.png"
ax.get_figure().savefig(fig_path, dpi = 400)


#分割图
fig = plt.figure(dpi=400)
ax = sns.violinplot(data=overall, x="model", y="pq", hue='dataset',
                inner="quart", linewidth=1
#                palette={"Yes": "b", "No": ".85"})
              )
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/desc_seg/pq.png"
ax.get_figure().savefig(fig_path, dpi = 400)

fig = plt.figure(dpi=400)
ax = sns.violinplot(data=overall, x="model", y="sq", hue='dataset',
                inner="quart", linewidth=1
#                palette={"Yes": "b", "No": ".85"})
              )
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/desc_seg/sq.png"
ax.get_figure().savefig(fig_path, dpi = 400)


fig = plt.figure(dpi=400)
ax = sns.violinplot(data=overall, x="model", y="dq", hue='dataset',
                inner="quart", linewidth=1
#                palette={"Yes": "b", "No": ".85"})
              )
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/desc_seg/dq.png"
ax.get_figure().savefig(fig_path, dpi = 400)



# 在不同 dist area cnt 上的表现
te['dist_cut'] = pd.qcut(te.dist, q=4 ,labels=['q1','q2','q3','q4'])
te['area_cut'] = pd.qcut(te.area, q=4 ,labels=['q1','q2','q3','q4'])
te['cnt_cut'] = pd.qcut(te.cnt, q=4 ,labels=['q1','q2','q3','q4'])

fig = plt.figure(dpi=400)
temp =te.groupby(['dist_cut','model']).mean().reset_index()
ax = sns.lineplot(data=temp, x="dist_cut", y="f1", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/dist.jpg"
# ax.get_figure().savefig(fig_path, dpi = 400)

fig = plt.figure(dpi=100)
temp =te.groupby(['area_cut','model']).median().reset_index()
ax = sns.lineplot(data=temp, x="area_cut", y="f1", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/area.jpg"
ax.get_figure().savefig(fig_path, dpi = 400)


fig = plt.figure(dpi=100)
temp =te.groupby(['cnt_cut','model']).median().reset_index()
ax = sns.lineplot(data=temp, x="cnt_cut", y="f1", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/cnt.pdf"
ax.get_figure().savefig(fig_path, dpi = 400)

fig = plt.figure(dpi=100)
temp =te.groupby(['dist_cut','model']).mean().reset_index()
ax = sns.lineplot(data=temp, x="dist_cut", y="pq", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/dist_pq.jpg"
ax.get_figure().savefig(fig_path, dpi = 400)

fig = plt.figure(dpi=100)
temp =te.groupby(['area_cut','model']).median().reset_index()
ax = sns.lineplot(data=temp, x="area_cut", y="pq", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/area_pq.jpg"
ax.get_figure().savefig(fig_path, dpi = 400)


fig = plt.figure(dpi=100)
temp =te.groupby(['cnt_cut','model']).median().reset_index()
ax = sns.lineplot(data=temp, x="cnt_cut", y="pq", hue='model')
sns.despine(left=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
fig_path = "/root/autodl-tmp/visualize/figure/seg/dist_area_cnt/cnt_pq.jpg"
ax.get_figure().savefig(fig_path, dpi = 400)