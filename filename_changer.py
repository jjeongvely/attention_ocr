import os


#filename_changer
# dir_path = "./test/new_total"
# idx = 0
#
# for path, dirs, files in os.walk(dir_path):
#     for file in files:
#         file_path = os.path.join(path, file)
#         new_filename = 'number_plates_{:02d}.png'.format(idx)
#         new_file_path = os.path.join(path, new_filename)
#
#         os.rename(file_path, new_file_path)
#         idx += 1

img_ls = []
dir_path = "/home/qisens/nanonets-ocr-sample-python/images"
for path, dirs, files in os.walk(dir_path):
    for file in files:
        file = file.split('.')[0]
        img_ls.append(file)

anno_ls = []
anno_dir_path = "/home/qisens/nanonets-ocr-sample-python/annotations/json"
for path,dirs,files in os.walk(anno_dir_path):
    for file in files:
        file = file.split('.')[0]
        anno_ls.append(file)

for anno in anno_ls:
    if not anno in img_ls:
        anno = '{}.json'.format(anno)
        anno_path = os.path.join(anno_dir_path, anno)
        os.remove(anno_path)
