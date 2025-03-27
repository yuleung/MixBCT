import mxnet as mx
import os
import numpy as np
import numbers
split_ratio = 0.3

root_dir = './'  #dataset path
path_imgrec = os.path.join(root_dir,  'train.rec')#'train_part_70_class.rec')#'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')#'train_part_70_class.idx')#'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
imgidx = np.array(list(imgrec.keys))
imgidx = imgidx[:5179511]

print('loaded!')
label_ = []
for i in range(1,len(imgidx)):
    s = imgrec.read_idx(i)
    head, img = mx.recordio.unpack(s)
    label = head.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    label_.append(label)
    #if i % 10000 == 0:
        #print(i)

total_class_num = len(set(label_))   #[0, total_class_num -1]
print(len(imgidx))
print(total_class_num)


per30class_class_num = int(round(total_class_num * split_ratio))  # < per30class_class_num
per70class_class_num = total_class_num - per30class_class_num

pre_label = 0
per30class_img_num = 0
for i in range(1,len(imgidx)):
    s = imgrec.read_idx(i)
    head, img = mx.recordio.unpack(s)
    label = head.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    if label < pre_label:
        print('label not countinue!')
    elif label > pre_label:
        per_label = label
    if label == per30class_class_num:
        per30class_img_num = i-1
        break

per70class_img_num = len(imgidx) - per30class_img_num  - 1

print('total_class_num: ', total_class_num)
print('per30class_class_num: ', per30class_class_num)
print('per70class_class_num: ', per70class_class_num)
print('total_imgidx: ', len(imgidx)-1)
print('per30class_img_num: ', per30class_img_num)
print('per70class_img_num: ', per70class_img_num)
write_record_per30_class = mx.recordio.MXIndexedRecordIO(f"./train_part_{int(split_ratio*100)}_class.idx","./train_part_{}_class.rec".format(int(split_ratio*100)), 'w')
write_record_per70_class = mx.recordio.MXIndexedRecordIO(f"./train_part_{int(100-split_ratio*100)}_class.idx","./train_part_{}_class.rec".format(int(100 - split_ratio*100)), 'w')


for i in range(len(imgidx)):
    if i == 0:
        s = imgrec.read_idx(i)
        P,_ = mx.recordio.unpack(s)
        header_30 = mx.recordio.IRHeader(flag=2, label=np.array([float(per30class_img_num+1), float(per30class_class_num)]), id=0, id2=0)
        s_30 = mx.recordio.pack(header_30,_)
        write_record_per30_class.write_idx(i,s_30)
        header_70 = mx.recordio.IRHeader(flag=2, label=np.array([float(per70class_img_num+1), float(per70class_class_num)]), id=0, id2=0)
        s_70 = mx.recordio.pack(header_70,_)
        write_record_per70_class.write_idx(i,s_70)

    elif i <= per30class_img_num:
        s = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(s)
        header =  mx.recordio.IRHeader(flag=2, label=np.array([header.label[0], header.label[1]]), id=i, id2=0)
        s = mx.recordio.pack(header,img)
        write_record_per30_class.write_idx(i,s)

    else:
        s = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(s)
        header =  mx.recordio.IRHeader(flag=2, label=np.array([header.label[0]-per30class_class_num, header.label[1]]), id=i-per30class_img_num, id2=0)
        s = mx.recordio.pack(header,img)
        write_record_per70_class.write_idx(i-per30class_img_num,s)

write_record_per30_class.close()
write_record_per70_class.close()
print('Done of class {}-{}_split!'.format(split_ratio*100, 100-split_ratio*100))


list_label_count = []
pre_label = 0
count = 0
for i in range(len(imgidx)):
    if i == 0:
        s = imgrec.read_idx(i)
        header,_ = mx.recordio.unpack(s)
    else:
        s = imgrec.read_idx(i)
        header,_ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        if label <= pre_label:
            count +=1
        else:
            pre_label = label
            list_label_count.append(count)
            count = 1
list_label_count.append(count)


print('len(list_label_count):{} == total_class_num:{}?'.format(len(list_label_count), total_class_num))

list_label30_count = []
list_label70_count = []
#per70_sub_class = 0
data30_img_count = 0
data70_img_count = 0
for i in list_label_count:
    if i == 1:
        data30_img_count += 1
        data70_img_count += 1
        list_label30_count.append(-1)
        list_label70_count.append(-1)
    else:
        data30_img_count += max(1, round(i * split_ratio))
        data70_img_count += i-max(1, round(i * split_ratio))
        list_label30_count.append(max(1, round(i * split_ratio)))
        list_label70_count.append(i-max(1, round(i * split_ratio)))

print('data30_img_count: ',data30_img_count)
print('data70_img_count: ',data70_img_count)





write_record_30_data = mx.recordio.MXIndexedRecordIO(f"./train_part_{int(split_ratio*100)}_data.idx",f"./train_part_{int(split_ratio*100)}_data.rec", 'w')
write_record_70_data = mx.recordio.MXIndexedRecordIO(f"./train_part_{int(100-split_ratio*100)}_data.idx",f"./train_part_{int(100-split_ratio*100)}_data.rec", 'w')

target = [-1]
target_id = [-1]
id_30 = 0
id_70 = 0
for i in range(len(list_label_count)):
    if list_label30_count[i] == -1:
        target.append(-1)
        target_id.append([id_30+1, id_70+1])
        id_30 += 1
        id_70 += 1          
    else:
        for j in range(list_label30_count[i]):
            target.append(0)
            target_id.append(id_30+j+1)
        id_30 += list_label30_count[i]
        for j in range(list_label70_count[i]):
            target.append(1)
            target_id.append(id_70+j+1)
        id_70 += list_label70_count[i]

print(sum(list_label_count))

count_img_data30 = 0
count_img_data70 = 0
for i in range(len(imgidx)):
    if i == 0:
        s = imgrec.read_idx(i)
        P,_ = mx.recordio.unpack(s)
        header_30 = mx.recordio.IRHeader(flag=2, label=np.array([float(data30_img_count+1), float(total_class_num)]), id=0, id2=0)
        s_30 = mx.recordio.pack(header_30,_)
        write_record_30_data.write_idx(i,s_30)
        header_70 = mx.recordio.IRHeader(flag=2, label=np.array([float(data70_img_count+1), float(total_class_num)]), id=0, id2=0)
        s_70 = mx.recordio.pack(header_70,_)
        write_record_70_data.write_idx(i,s_70)
    else:
        s = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(s)
        #print(header.label)
        if target[i] == 0:
            header = mx.recordio.IRHeader(flag=2, label=np.array([header.label[0], header.label[1]]), id=target_id[i], id2=0)
            count_img_data30 += 1
            s_30 = mx.recordio.pack(header, img)
            write_record_30_data.write_idx(target_id[i], s_30)
        elif target[i] == 1:
            header = mx.recordio.IRHeader(flag=2, label=np.array([header.label[0], header.label[1]]), id=target_id[i], id2=0)
            count_img_data70 += 1
            s_70 = mx.recordio.pack(header, img)
            write_record_70_data.write_idx(target_id[i], s_70)
        elif target[i] == -1:
            header = mx.recordio.IRHeader(flag=2, label=np.array([header.label[0], header.label[1]]), id=target_id[i][0], id2=0)
            count_img_data30 += 1                                                                                                                                   
            s_30 = mx.recordio.pack(header, img)
            header = mx.recordio.IRHeader(flag=2, label=np.array([header.label[0], header.label[1]]), id=target_id[i][1], id2=0)
            count_img_data70 += 1                                                                                                                                   
            s_70 = mx.recordio.pack(header, img)
            write_record_30_data.write_idx(target_id[i][0], s_30)
            write_record_70_data.write_idx(target_id[i][1], s_70)       
        else:
            print('Error!')
            exit(-1)


write_record_30_data.close()
write_record_70_data.close()
print('count_img_data30: ', count_img_data30)
print('count_img_data70: ', count_img_data70)


print('Done of data {}-{}_split!'.format(split_ratio*100, 100-split_ratio*100))
