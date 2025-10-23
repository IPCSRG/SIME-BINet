
from openpyxl import Workbook

# 创建一个Workbook对象，这将会创建一个Excel文件
wb = Workbook()

# 获取当前激活的worksheet
ws = wb.active

# 修改worksheet的名称
ws.title = "MySheet"

G = {}
seg = {}
with open(r"D:\Documents\Method_all\Mothod20241107\checkpoints\Muals_mask_places2_400_trans_inpudmasked_addmasked__first_2-3\loss_log.txt", "r") as file:
    content = file.readlines()

for line in content:
    if 'G' in line:
        a = line.split(', ')
        epoch = a[0].split('epoch: ')[1]
        g = line.split('G: ')[1].split(' G_content:')[0]
        sg = line.split('G_seg: ')[1].split(' D_real:')[0]
        if str(epoch) in G.keys():
            tt = G[str(epoch)]
            tt.append(float(g))
            print(1)
        else:
            G[str(epoch)] = [float(g)]


        if str(epoch) in seg.keys():
            tt = seg[str(epoch)]
            tt.append(float(sg))
            print(1)
        else:
            seg[str(epoch)] = [float(sg)]

print(G)
print(seg)
g_list = []
seg_list = []
for i in range(1, 401):
    g_score = G[str(i)]
    g_list.append(sum(g_score)/len(g_score))
    seg_score = seg[str(i)]
    seg_list.append(sum(seg_score) / len(seg_score))
    print(111)
print(11)

# 将数据写入worksheet
for i in range(400):
    ws.append([g_list[i],seg_list[i]])

# 保存Excel文件
wb.save("example.xlsx")