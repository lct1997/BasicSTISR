import csv
import glob
import math

import torch
import os
import datetime
import pickle
import copy

from tensorboardX import SummaryWriter
from tqdm import tqdm

from loss.image_loss import SemanticLoss
from utils import util, ssim_psnr
from IPython import embed
import torch.nn.functional as F
import cv2
from interfaces import base
from utils.img_utils import torch_rotate_img
from utils.metrics import get_string_aster, get_string_crnn, Accuracy,get_string_abinet,get_string_parseq
from utils.util import str_filt
import numpy as np
from ptflops import get_model_complexity_info
import editdistance
import time
import logging
from dataset.dataset import Lable2Tensor
from torch import optim as optim
import lpips
import setup
tri_ssim = ssim_psnr.TRI_SSIM()
sem_loss = SemanticLoss()
abi_charset = setup.CharsetMapper()
label2tensor = Lable2Tensor()
ssim = ssim_psnr.SSIM()
lpips_vgg = lpips.LPIPS(net="vgg")

class TextSR(base.TextBase):
    def loss_stablizing(self, loss_set, keep_proportion=0.7):
        # acsending
        sorted_val, sorted_ind = torch.sort(loss_set)
        batch_size = loss_set.shape[0]
        # print("batch_size:", loss_set, batch_size)
        loss_set[sorted_ind[int(keep_proportion * batch_size)]:] = 0.0
        return loss_set

    def model_inference(self, images_lr, model_list, text_emb=None):
        ret_dict = {}
        ret_dict['duration']= 0.
        images_sr = []
        before = time.time()
        if text_emb is not None:  # ["tatt", "tbsrn", "tpgsr"]有文本先验和mask
            output = model_list[0](images_lr, text_emb.detach())
        elif self.args.arch in ["tg"]: # 没有mask
            output = model_list[0](images_lr[:,:3,:,:])
        else:
            output = model_list[0](images_lr)
        after = time.time()
        # print("fps:", (after - before))
        ret_dict["duration"] += (after - before)
        if type(output)==list:
            images_sr.append(output[0])
        else:
            images_sr.append(output)
        ret_dict["images_sr"] = images_sr
        return ret_dict



    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        model_list = [model]
        # -------- tensorboard logger -------------------------
        tensorboard_dir = os.path.join('tb_logger',self.args.arch)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir,exist_ok=True)
        elif self.args.resume is None:
            print("Directory exist, remove events...")
            for file in glob.glob(tensorboard_dir + "/*"):
                os.remove(file)
        self.results_recorder = SummaryWriter(tensorboard_dir)
        # ------- init text recognizer for eval here --------
        aster, aster_info = self.CRNN_init()
        aster.eval()

        test_bible = {}
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init() # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }
        elif self.args.test_model == "MORAN":
            moran, aster_info = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        if self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
            recognizer_path = os.path.join('experiments', self.args.arch, "recognizer_best.pth")  # 这个是训练过程中生成的模型
            if os.path.isfile(recognizer_path):
                aster_student, aster_stu_info = self.CRNN_init(recognizer_path=recognizer_path)
            else:
                aster_student, aster_stu_info = self.CRNN_init()
            aster_student.train()
            optimizer_G = self.optimizer_init_t(model_list[0], recognizer=aster_student)
        else:
            optimizer_G = self.optimizer_init(model_list[0])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, 400,0.5)

        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        log_path = os.path.join('experiments', self.args.arch, "eval.csv")

        for model in model_list:
            model.train()

        for epoch in range(cfg.epochs):
            # ----------------start training here -------------------
            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1
                for model in model_list:
                    for p in model.parameters():
                        p.requires_grad = True

                images_hr, _, images_lr, _, _, label_strs, label_vecs, _, weighted_tics = data
                images_lr = images_lr.to(self.device) # [b,4,16,64]
                images_hr = images_hr.to(self.device) # [b,4,32,128]
                loss_img = 0.
                # ----------------------- 该部分是 tatt 用到的 ----------------------------
                if self.args.arch in ["tatt","tpgsr","tbsrn"]:
                    batch_size = images_lr.shape[0]
                    angle_batch = np.random.rand(batch_size) * 5 * 2 - 5
                    arc = angle_batch / 180. * math.pi
                    rand_offs = torch.tensor(np.random.rand(batch_size)).float()
                    arc = torch.tensor(arc).float()
                    images_lr = torch_rotate_img(images_lr, arc, rand_offs)
                    images_hr = torch_rotate_img(images_hr, arc, rand_offs)
                    images_lr_ret = torch_rotate_img(images_lr.clone(), -arc, rand_offs)
                    # ------------------- 识别LR -----------------------------
                    cascade_images = images_lr
                    cascade_images = cascade_images.detach()
                    aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :])
                    label_vecs_logits = aster_student(aster_dict_lr)  #需要训练的识别网络
                    label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                    label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                    # ------------------ 识别HR ------------------------------
                    aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                    label_vecs_logits_hr = aster(aster_dict_hr)  #不需要训练
                    label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1).detach()
                """ 
                网络输入与输出 
                """
                # ------------------ 两种传统超分方法 + 四种用于文本图像的方法 ----------------------------
                if self.args.arch in ["srcnn", "srres", "tsrn"]:
                    cascade_images = model_list[0](images_lr)
                    loss = image_crit(cascade_images, images_hr)
                elif self.args.arch in ["tg"]:
                    cascade_images = model_list[0](images_lr[:,:3,:,:])
                    loss, mse_loss, attention_loss, recognition_loss = image_crit(cascade_images[:,:3,:,:], images_hr[:,:3,:,:], label_strs)
                elif self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
                    cascade_images = model_list[0](cascade_images, label_vecs_final.detach())
                    loss_tp = sem_loss(label_vecs, label_vecs_hr)   # TP损失：LR和HR的识别结果损失
                    loss = image_crit(cascade_images, images_hr)  # L2损失：SR和HR的图像损失
                    if self.args.arch == "tatt":
                        """ TPGSR没有这个损失 """
                        cascade_images_sr_ret = model_list[0](images_lr_ret, label_vecs_final.detach())  # 获得旋转后的SR
                        cascade_images_ret = torch_rotate_img(cascade_images_sr_ret, arc, rand_offs)  # 再转回来
                        loss_tsc = (1 - tri_ssim(cascade_images_ret, cascade_images, images_hr).mean()) * 0.1  #TSC损失
                        loss += loss_tsc
                    loss += loss_tp
                loss_img_each  =  loss.mean() * 100  # 总损失
                loss_img += loss_img_each
                loss_im = loss_img
                optimizer_G.zero_grad()
                loss_im.backward()
                for model in model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()
                # ------------------tensorboard record -----------------------------
                if iters % 5 == 0:
                    self.results_recorder.add_scalar('loss/total_loss', float(loss_im.data), global_step=iters)
                if iters % cfg.displayInterval == 0:
                    logging.info('[{}]\t'
                                 'Epoch: [{}][{}/{}]\t'
                                 'total_loss {:.3f}\t'
                                 .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         epoch, j + 1, len(train_loader),
                                         float(loss_im.data)
                                         ))

                if iters % cfg.VAL.valInterval == 0:
                    logging.info('======================================================')
                    current_acc_dict = {}
                    psnr_dict={}
                    ssim_dict={}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)
                        for model in model_list:
                            model.eval()
                            for p in model.parameters():
                                p.requires_grad = False
                        if self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
                            """ 验证时不求导 """
                            aster_student.eval()
                            for p in aster_student.parameters():
                                p.requires_grad = False

                            metrics_dict = self.eval(
                                model_list,
                                val_loader,
                                image_crit,
                                iters,
                                [test_bible[self.args.test_model], aster_student, aster],
                                aster_info,
                                data_name
                            )
                            for p in aster_student.parameters():
                                p.requires_grad = True
                            aster_student.train()
                        else:
                            # text_recognizer = crnn
                            metrics_dict = self.eval(
                                model_list,
                                val_loader,
                                image_crit,
                                iters,
                                [test_bible[self.args.test_model], None, None],
                                aster_info,
                                data_name
                            )
                        psnr_dict[data_name]=metrics_dict['psnr_avg']
                        ssim_dict[data_name]=metrics_dict['ssim_avg']
                        """ 验证完成后进行求导 """
                        for p in aster_student.parameters():
                            p.requires_grad = True
                        aster_student.train()
                        for model in model_list:
                            for p in model.parameters():
                                p.requires_grad = True
                            model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        for key in metrics_dict:
                            if key in ["psnr_avg", "ssim_avg", "accuracy"]:
                                self.results_recorder.add_scalar('eval/' + key + "_" + data_name,
                                                                 float(metrics_dict[key]),
                                                                 global_step=iters)
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            # logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))
                            logging.info('best_%s = %.2f%% (A New Record)' % (data_name, best_history_acc[data_name] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow(
                                    [epoch, data_name, metrics_dict['accuracy'], metrics_dict['psnr_avg'],
                                     metrics_dict['ssim_avg'], "best_{}".format(data_name)])
                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                            with open(log_path, "a+", newline="") as out:
                                writer = csv.writer(out)
                                writer.writerow(
                                    [epoch, data_name, metrics_dict['accuracy'], metrics_dict['psnr_avg'],
                                     metrics_dict['ssim_avg']])

                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        logging.info('avg_acc {:.4f}%'.format(100*(current_acc_dict['easy']*1619
                                                              +current_acc_dict['medium']*1411
                                                              +current_acc_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('==================')
                        logging.info('bset psnr {:.4f}'.format(1*(psnr_dict['easy']*1619
                                                              +psnr_dict['medium']*1411
                                                              +psnr_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('best ssim {:.4f}'.format(1 * (ssim_dict['easy'] * 1619
                                                                + ssim_dict['medium'] * 1411
                                                                + ssim_dict['hard'] * 1343) / (1343 + 1411 + 1619)))
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer=aster_student)
                        with open(log_path, "a+", newline="") as out:
                            writer = csv.writer(out)
                            writer.writerow([epoch, "****", "*****", "******", "******", "******", "best_sum",
                                             "avg_acc {:.4f}%".format(100*(current_acc_dict['easy']*1619
                                                              +current_acc_dict['medium']*1411
                                                              +current_acc_dict['hard']*1343)/(1343+1411+1619))])

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=aster_student)

            lr_scheduler.step()

    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info, data_name=None):
        # ------- 验证时，设定参数不求导，可以省内存 ------------------------
        for model in model_list:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        aster[0]["model"].eval()
        for p in aster[0]["model"].parameters():
            p.requires_grad = False
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
                       'psnr_lr': [],
                       'ssim_lr': [],
                       'cnt_psnr_lr': [],
                       'cnt_ssim_lr': [],
                       'psnr': [],
                       'ssim': [],
                       'cnt_psnr': [],
                       'cnt_ssim': [],
                       'accuracy': 0.0,
                       'psnr_avg': 0.0,
                       'ssim_avg': 0.0,
                        'edis_LR': [],
                        'edis_SR': [],
                        'edis_HR': [],
                        'LPIPS_VGG_LR': [],
                        'LPIPS_VGG_SR': []
                       }

        counters = {0: 0}
        image_counter = 0
        rec_str = ""
        sr_infer_time = 0

        for i, data in (enumerate(tqdm(val_loader))):
            images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt, _ = data
            # print("label_strs:", label_strs)
            images_lr = images_lrraw.to(self.device)
            images_hr = images_hrraw.to(self.device)
            val_batch_size = images_lr.shape[0]

             # ----------------------- 该部分是 tatt 和 tpgsr用到的文本先验生成器 ----------------------------
            if self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
                cascade_images = images_lr
                aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :])
                aster_crnn_lr = aster[1](aster_dict_lr) #用学习过的识别器来进行先验生成，0==>可学习，1==>不学习
                label_vecs = torch.nn.functional.softmax(aster_crnn_lr, -1)  #[26,48,37] ==> [26,48,37]
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2) #[48,37,1,26]
                ret_dict = self.model_inference(cascade_images, model_list, label_vecs_final) # get SR
            else:
                ret_dict = self.model_inference(images_lr, model_list)
            sr_infer_time += ret_dict["duration"]
            images_sr = ret_dict["images_sr"]

            # == 首先解析输入图像为文本识别器的输入形式 =======
            aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
            aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])
            # ==== 之后是将解析后的图像用于文本识别  这里只对LR和HR进行识别====
            if self.args.test_model == "MORAN":
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)
            # ============对SR进行图像解析和通过文本识别器 ==============
            if type(images_sr) == list:
                predict_result_sr = []
                image = images_sr[0]
                aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr_ = aster[0]["string_process"](aster_output_sr)
                elif self.args.test_model == "ASTER":
                    predict_result_sr_, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]
                predict_result_sr.append(predict_result_sr_)

                img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(img_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            else:
                aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr) # 对超分结果进行识别
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr = aster[0]["string_process"](aster_output_sr)
                elif self.args.test_model == "ASTER":
                    predict_result_sr, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]

                img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")
                metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(images_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            if self.args.test_model == "CRNN":#之前是只对SR的识别结果进行的处理，这里将HR和LR同样进行处理
                predict_result_lr = aster[0]["string_process"](aster_output_lr)
                predict_result_hr = aster[0]["string_process"](aster_output_hr)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                predict_result_hr = [pred.split('$')[0] for pred in sim_preds]
            filter_mode = 'lower'

            for batch_i in range(images_lr.shape[0]):
                label = label_strs[batch_i]
                image_counter += 1
                rec_str += str(image_counter) + ".jpg," + label + "\n"

                if str_filt(predict_result_sr[0][batch_i], filter_mode) == str_filt(label, filter_mode):
                    counters[0] += 1
                if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_lr += 1
                if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_hr += 1
            if self.args.test is True:
                self.test_display(images_lr[:, :3, :], images_sr[0][:, :3, :], images_hr[:, :3, :], predict_result_lr,
                                  predict_result_sr[0], label_strs)
            sum_images += val_batch_size
            torch.cuda.empty_cache()

        # 已经把整个测试集跑完
        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['psnr']) + 1e-10)

        psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)
        lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
        lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

        logging.info('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              'LPIPS {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr)))

        logging.info('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              'LPIPS_LR {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr)))

        # logging.info('save display images')
        if self.args.display:
            logging.info('Save eval images')
            self.tripple_display(images_lr[:, :3, :], images_sr[0][:, :3, :], images_hr[:, :3, :],
                                 predict_result_lr, predict_result_sr[0], label_strs)
        accuracy = round(counters[0] / sum_images, 4)# loader已经跑完了，把全部的测试集图片和全部正确个数相除

        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)


        logging.info('sr_accuray_iter' + ': %.2f%%' % (accuracy * 100))


        logging.info('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        logging.info('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        inference_time = sum_images / sr_infer_time
        logging.info("AVG inference:{}".format(inference_time))
        logging.info("sum_images:{}".format(sum_images))

        return metric_dict


    def test(self):
        total_acc = {}
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        model_list = [model]
        logging.info('Using text recognizer {}'.format(self.args.test_model))
        test_bible = {}

        aster, aster_info = self.CRNN_init()
        aster_info = None
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()  # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }
        elif self.args.test_model == "MORAN":
            moran, aster_info = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        if self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
            recognizer_path = os.path.join('experiments', self.args.arch, "recognizer_best.pth")  # 这个是训练过程中生成的模型
            if os.path.isfile(recognizer_path):
                aster_student, aster_info = self.CRNN_init(recognizer_path=recognizer_path)
            else:
                aster_student, aster_info = self.CRNN_init(recognizer_path=None)
            aster_student.train()
            for p in aster_student.parameters():
                p.requires_grad = True

        for k, val_loader in enumerate(val_loader_list):
            data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            logging.info('testing %s' % data_name)
            for model in model_list:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
            if self.args.arch in ["tatt", "tpgsr", "tbsrn"]:
                aster_student.eval()
                for p in aster_student.parameters():
                    p.requires_grad = False

                metrics_dict = self.eval(
                    model_list,
                    val_loader,
                    image_crit,
                    0,
                    [test_bible[self.args.test_model], aster_student, aster],
                    aster_info,
                    data_name
                )
            else:
                metrics_dict = self.eval(
                    model_list,
                    val_loader,
                    image_crit,
                    0,
                    [test_bible[self.args.test_model], None, None],
                    aster_info,
                    data_name
                )
            acc = float(metrics_dict['accuracy'])
            total_acc[data_name]=acc
            logging.info('best_%s = %.2f%%' % (data_name, acc * 100))
        logging.info('avg_acc(Easy,Medium,Hard) is {:.3f}'.format(100*(total_acc['easy']*1619
                                                              +total_acc['medium']*1411
                                                              +total_acc['hard']*1343)/(1343+1411+1619)))
        logging.info('Test with recognizer {} finished!'.format(self.args.test_model))
