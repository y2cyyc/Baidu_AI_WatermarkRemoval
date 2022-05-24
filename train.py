import paddle
from loss_function import PSNRLoss, SSIMLoss
from model.get_model import get_unetrw, get_unetrw2, get_res50, get_unet_naf, get_unet_ca, get_mw_nafNet, get_ridnet, get_unet, get_swim
from dataloader2 import MyDateset
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
paddle.set_device('gpu')
batch_s = 8
output_model_dir = './rid'
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)

# log file
file_path = './rid/train_log.txt'
f_log = open(file_path, 'a')

######## model
# model = get_unetrw()
model = get_swim()
# model = get_unet_naf()
model.train()

train_dataset=MyDateset()

# 需要接续之前的模型重复训练可以取消注释
# param_dict = paddle.load('./rid/model_27.pdparams')
# model.load_dict(param_dict)
# model_pretrain_dict = paddle.load('./models_para/model_19.pdparams')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in model_pretrain_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_dict(model_dict)

train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_s,
    shuffle=True,
    drop_last=False)

losspsnr = PSNRLoss()
lossfn = SSIMLoss(window_size=3,data_range=1)

max_epoch=50
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.0008, T_max=max_epoch)
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

start_epoch = 0
epoch_steps = train_dataset.__len__()//batch_s
print('epoch_steps', epoch_steps)
now_step = start_epoch*epoch_steps

for epoch in range(start_epoch, max_epoch):
    f_log.write('epoch:'+str(epoch)+'\n')
    for step, data in enumerate(train_dataloader):
        now_step+=1

        img, label = data
        pre = model(img)

        loss1 = lossfn(pre,label).mean()
        loss2 = losspsnr(pre,label).mean()
        loss = (loss1+loss2/100)/2
        # loss = loss2/100

        loss.backward()
        opt.step()
        opt.clear_gradients()

        scheduler.step(epoch + step/epoch_steps)

        lr_update = opt.get_lr()
        if now_step%10==0:
            content_log = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' ' + "epoch: {}, batch: {}, loss is: {}, lr:{}".format(epoch, step, loss.mean().numpy(), lr_update)
            print(content_log)
            f_log.write(content_log + '\n')
    paddle.save(model.state_dict(), os.path.join(output_model_dir, 'model_' + str(epoch)+'.pdparams'))
f_log.close()
paddle.save(model.state_dict(), 'model.pdparams')

