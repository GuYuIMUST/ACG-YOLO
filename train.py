import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('E:/ACG-YOLO/ultralytics/cfg/models/12/ACG-YOLO.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='E:/ACG-YOLO/GDXray/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                # device='0,1', 
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
