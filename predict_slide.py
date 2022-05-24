# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2
from model.model_u import UNet
import paddle
import numpy as np
import math


def process(src_image_dir, save_dir):

    model = UNet()
    # model = swim_fpn2()
    model_statedict = paddle.load('./model/model_006.pdparams')
    # model_statedict = paddle.load('./model/model_37.pdparams')
    model.set_state_dict(model_statedict)

    model.eval()


    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        # do something
        img = cv2.imread(image_path)
        H, W, C = img.shape
        B = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = paddle.vision.transforms.resize(img, (512, 512), interpolation='bilinear')
        img = img.transpose((2, 0, 1))
        img = img / 255
        img = paddle.to_tensor(img).astype('float32')
        img = paddle.unsqueeze(img, 0)

        crop_size = 512
        stride = 448
        seg_pred = paddle.zeros([B, 3, H, W])
        count_pred = paddle.zeros([B, 3, H, W])
        num_h = math.ceil((H - crop_size) / stride) + 1
        num_w = math.ceil((W - crop_size) / stride) + 1

        with paddle.no_grad():
            for i_h in range(num_h):
                for i_w in range(num_w):
                    start_h = i_h * stride
                    end_h = start_h + crop_size
                    start_w = i_w * stride
                    end_w = start_w + crop_size

                    if end_h > H:
                        end_h = H
                        start_h = max(H - crop_size, 0)
                    if end_w > W:
                        end_w = W
                        start_w = max(W - crop_size, 0)
                    patch = img[:, :, start_h:end_h, start_w:end_w]

                    s_patch_predict = model(patch)

                    seg_pred[:, :, start_h:end_h, start_w:end_w] += s_patch_predict
                    count_pred[:, :, start_h:end_h, start_w:end_w] += 1

            out_image = seg_pred / count_pred


            out_image = paddle.squeeze(out_image, 0)
            out_image = out_image.cpu().detach().numpy().transpose(1, 2, 0)

            out_image = (out_image * 255).astype('uint8')
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)


            # 保存结果图片
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, out_image)


if __name__ == "__main__":
    # assert len(sys.argv) == 3

    # src_image_dir = sys.argv[1]
    # save_dir = sys.argv[2]

    src_image_dir = r'D:\yyc\competition\AIstudio\WatermarkRemoval\B_testData'
    save_dir = r'D:\yyc\competition\AIstudio\WatermarkRemoval\results\result'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)