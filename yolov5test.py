# git clone https://github.com/ultralytics/yolov5.git
# cd yolov5
# pip install -qr requirements.txt

import os
from glob import glob

# root_dir = "C:\content\export"
# img_dir = os.path.join(root_dir, "images")
# label_dir = os.path.join(root_dir,"labels")

root_dir = "C:\content\export"
img_dir = os.path.join(root_dir+r'\images', "K-039147")
label_dir = os.path.join(root_dir+r'\labels',"K-039147")

data = glob(os.path.join(img_dir,"*.png"))
data

train = data[:910]
valid = data[910:1290]
test = data[1290:]

# train.txt
with open(os.path.join(root_dir, "train.txt"), 'w') as f:
	f.write('\n'.join(train) + '\n')

# valid.txt
with open(os.path.join(root_dir, "valid.txt"), 'w') as f:
	f.write('\n'.join(valid) + '\n')

# test.txt
with open(os.path.join(root_dir, "test.txt"), 'w') as f:
	f.write('\n'.join(test) + '\n')

#----------------------------------------------------------------------------# 
# train.txt에 추가할 경우(append)
with open(os.path.join(root_dir, "train.txt"), 'a') as f:
	f.write('\n'.join(train) + '\n')
 
# valid.txt 추가할 경우
with open(os.path.join(root_dir, "valid.txt"), 'a') as f:
	f.write('\n'.join(valid) + '\n')

# test.txt 추가할 경우
with open(os.path.join(root_dir, "test.txt"), 'a') as f:
	f.write('\n'.join(test) + '\n') 
#----------------------------------------------------------------------------#
 
 
import yaml

yaml_data = {"names":['듀카브정30/5밀리그램', '듀카브정30/10밀리그램'], # 클래스 이름
             "nc":2, # 클래스 수
             "path":root_dir, # root 경로
             "train":os.path.join(root_dir, "train.txt"), # train.txt 경로
             "val":os.path.join(root_dir, "valid.txt"), # valid.txt 경로
             "test":os.path.join(root_dir, "test.txt") # test.txt 경로
             }

with open(os.path.join(root_dir, "custom.yaml"), "w") as f:
  yaml.dump(yaml_data, f)
  

  
# 학습
# python train.py --img 640 --batch 32 --epochs 10 --device 0 --data /content/export/custom.yaml --weights yolov5s.pt --name dukarb1.test 
# 검증 
# python detect.py --weights /content/export/yolov5/runs/train/medi.test/weights/best.pt --source /content/export/test1.png
# python detect.py --weights /content/export/yolov5/runs/train/sidukarb.test/weights/best.pt --source /content/export/test10+5.png 
# 테스트
# python val.py --task "test" --data /content/export/custom.yaml --weights /content/export/yolov5/runs/train/dukarb.test4/weights/best.pt --device 0 --save-txt


# gc 확인, cuda cache 정리
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
