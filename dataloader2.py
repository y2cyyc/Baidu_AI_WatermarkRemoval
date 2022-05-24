import paddle
import os
import numpy as np
import cv2
import random


class MyDateset(paddle.io.Dataset):
    def __init__(self, mode='train', patch=512,
                 watermark_dir='D:/yyc/competition/AIstudio/WatermarkRemoval/watermark_dataset/watermark_datasets.part/',
                 bg_dir='D:/yyc/competition/AIstudio/WatermarkRemoval/watermark_dataset/bg_images/'):
        super(MyDateset, self).__init__()
        # 1->bg:
        # self.part_num = [1, 2, 3, 10]
        # self.bg_num = np.arange(0,552).tolist() + np.arange(1657, 1841).tolist()
        self.bg_num = np.arange(0, 1841).tolist()

        self.mode = mode
        self.watermark_dir = watermark_dir
        self.bg_dir = bg_dir
        self.patch_size = patch

    def __getitem__(self, index):
        bg_select = index % len(self.bg_num)
        bg_item = self.bg_num[bg_select]

        # 读入bg文件
        # bg_image_00000.jpg , bg 文件的命名方式
        label = cv2.imread(os.path.join(self.bg_dir, 'bg_image_' + str(bg_item).zfill(5) + '.jpg'))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        H, W, _ = label.shape
        # label = paddle.vision.transforms.resize(label, (512,512), interpolation='bilinear')
        label = label.transpose((2, 0, 1))
        label = label / 255
        label = paddle.to_tensor(label).astype('float32')

        # 判断bg图片对应的水印文件所在的文件夹编号
        # if bg_select<184:
        #     part_index = self.part_num[0]
        # elif bg_select<368:
        #     part_index = self.part_num[1]
        # elif bg_select<552:
        #     part_index = self.part_num[2]
        # else:
        #     part_index = self.part_num[3]

        # 调整水印文件所在文件夹
        # watermark_dir_part = self.watermark_dir + str(part_index)
        watermark_dir_part = self.watermark_dir
        # print('watermark_dir_part', watermark_dir_part)

        # 随机选择20张水印文件作为网络输入
        watermark_num = np.random.randint(1, 531)

        # bg_image_00000_0001.jpg 水印文件的命名方式
        water_marker_list = []
        for k in range(0, 1):
            img = cv2.imread(os.path.join(watermark_dir_part,
                                          'bg_image_' + str(bg_item).zfill(5) + '_' + str(watermark_num + k).zfill(
                                              4) + '.jpg'))
            if img is None:
                print(os.path.join(watermark_dir_part,
                                   'bg_image_' + str(bg_item).zfill(5) + '_' + str(watermark_num + k).zfill(
                                       4) + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 0)
            img = img / 255
            img = paddle.to_tensor(img).astype('float32')
            water_marker_list.append(img)
        water_marker_input = paddle.concat(water_marker_list, 0)

        wi = random.randint(0, (W - self.patch_size))
        hi = random.randint(0, (H - self.patch_size))

        water_marker_input = water_marker_input[:, :, hi: hi + self.patch_size, wi: wi + self.patch_size].squeeze(0)
        label = label[:, hi: hi + self.patch_size, wi: wi + self.patch_size]

        water_marker_input = paddle.vision.transforms.resize(water_marker_input, (256, 256), interpolation='bilinear')
        label = paddle.vision.transforms.resize(label, (256, 256), interpolation='bilinear')

        return water_marker_input, label

    def __len__(self):
        return len(self.bg_num)


def main():
    train_dataset = MyDateset()
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=False)

    for step, data in enumerate(train_dataloader):
        img, label = data
        print(step, img.shape, label.shape)


# 对dataloader进行测试
if __name__ == '__main__':
    main()
