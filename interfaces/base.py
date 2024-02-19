import datetime
import logging
import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
import string
from PIL import Image
import torchvision
from torchvision import transforms
from collections import OrderedDict

from loss import image_loss, stroke_focus_loss
import ptflops
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset_real,alignCollate_realWTL, alignCollate_realWTLAMask
from utils.labelmaps import get_vocabulary, labels2strs
from model import recognizer, tsrn, tbsrn, tatt, srcnn, srresnet, tg, tpgsr
sys.path.append('../')
from utils import ssim_psnr, utils_moran, utils_crnn, str_filt
import dataset.dataset as dataset
import shutil


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.align_collate = alignCollate_realWTLAMask
        self.load_dataset = lmdbDataset_real
        self.align_collate_val = alignCollate_realWTL
        self.load_dataset_val = lmdbDataset_real

        # self.resume = config.TRAIN.resume
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.resume_train = args.resume_train
        self.batch_size = self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
            'chinese': open("al_chinese.txt", "r",encoding='UTF-8').readlines()[0].replace("\n", "")
        }
        self.test_data_dir = self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.config.TRAIN.VAL.vis_dir
        # self.ckpt_path = os.path.join('ckpt', self.vis_dir)
        self.ckpt_path = os.path.join('experiments', self.args.arch)
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.cal_psnr_weighted = ssim_psnr.weighted_calculate_psnr
        self.cal_ssim_weighted = ssim_psnr.SSIM_WEIGHTED()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len,
                                      rotate=False,
                                      test=False
                ))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=True),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN

        test_dataset = self.load_dataset_val(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         rotate=False
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=False),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self):
        cfg = self.config.TRAIN
        resume = self.resume

        # ----------------------- 定义模型和损失 -------------------------
        if self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb)
            image_crit = image_loss.ImageLoss(loss_weight=[1, 1e-4])
        elif self.args.arch == 'tg':
            model = tg.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb)
            image_crit = stroke_focus_loss.StrokeFocusLoss()
        elif self.args.arch == 'tpgsr':
            model = tpgsr.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb)
            image_crit = image_loss.ImageLoss(loss_weight=[1, 1e-4])
        elif self.args.arch == 'tbsrn':
            model = tbsrn.TBSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb)
            image_crit = image_loss.ImageLoss(loss_weight=[1, 1e-4])
        elif self.args.arch == 'tatt':
            model = tatt.TSRN_TL_TRANS(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb)
            image_crit = image_loss.ImageLoss(loss_weight=[1, 1e-4])
        model = model.to(self.device)
        image_crit.to(self.device)
        if cfg.ngpu > 1:
            print("multi_gpu", self.device)
            model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))
        if resume != '' :
            if self.resume_train is True:
                logging.info('loading pre-trained STISR model from %s ' % resume)
                if self.config.TRAIN.ngpu == 1:
                    if os.path.isdir(resume):
                        print("resume:", resume)
                        model_dict = torch.load(resume)['state_dict_G']
                        model.load_state_dict(model_dict, strict=False)
                    else:
                        print("resume:", resume)
                        loaded_state_dict = torch.load(resume)['state_dict_G']
                        model.load_state_dict(loaded_state_dict,strict=False)
                else:
                    print("resume:", resume)
                    model_dict = torch.load(resume)['state_dict_G']

                    if os.path.isdir(resume):
                        model.load_state_dict(
                            {'module.' + k: v for k, v in model_dict.items()}
                            , strict=False)
                    else:
                        model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(resume)['state_dict_G'].items()})
            else:
                logging.info('loading pre-trained STISR model from %s ' % resume)
                if self.config.TRAIN.ngpu == 1:
                    if os.path.isdir(resume):
                        print("resume:", resume)
                        model_dict = torch.load(resume)['state_dict_G']
                        model.load_state_dict(model_dict, strict=True)
                    else:
                        print("resume:", resume)
                        loaded_state_dict = torch.load(resume)['state_dict_G']
                        model.load_state_dict(loaded_state_dict, strict=True)
                else:
                    print("resume:", resume)
                    model_dict = torch.load(resume)['state_dict_G']

                    if os.path.isdir(resume):
                        model.load_state_dict(
                            {'module.' + k: v for k, v in model_dict.items()}
                            , strict=False)
                    else:
                        model.load_state_dict(
                            {'module.' + k: v for k, v in torch.load(resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "position_prior" not in n and p.requires_grad]}, # origin
            {
                "params": [p for n, p in model.named_parameters() if "position_prior" in n and p.requires_grad],  # fine tune
                "lr": self.args.lr_position,
            },
        ]
        optimizer = optim.Adam(param_dicts, lr=cfg.lr,
                               betas=(cfg.beta1, 0.999))
        return optimizer

    def optimizer_init(self, model, recognizer=None):
        cfg = self.config.TRAIN
        if not recognizer is None:
            if type(recognizer) == list:
                rec_params = []
                model_params = []
                for recg in recognizer:
                    rec_params += list(recg.parameters())
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                optimizer = optim.Adam(model_params + rec_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
            else:
                model_params = []
                if type(model) == list:
                    for m in model:
                        model_params += list(m.parameters())
                else:
                    model_params = list(model.parameters())
                optimizer = optim.Adam(model_params + list(recognizer.parameters()), lr=cfg.lr,betas=(cfg.beta1, 0.999))
        else:
            model_params = []
            if type(model) == list:
                for m in model:
                    model_params += list(m.parameters())
            else:
                model_params = list(model.parameters())
            optimizer = optim.Adam(model_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))

        return optimizer

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs):
        for i in (range(min(image_in.shape[0], self.config.TRAIN.VAL.n_vis))):
            tensor_in = image_in[i][:3, :, :]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3, :, :]
            tensor_target = image_target[i][:3, :, :]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join("experiments", self.args.arch, 'eval_vis', self.args.test_model)
            if not os.path.exists(out_root):
                os.makedirs(out_root, exist_ok=True)
            im_name = str_filt(pred_str_lr[i], 'lower') + '_' + str_filt(pred_str_sr[i], 'lower') + '_' + str_filt(label_strs[i], 'lower') + '_.png'
            im_name = im_name.replace('/', '')
            torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                # if (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):  #当预测的SR文字跟实际HR标签不同的时候，才进行保存
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]),
                                           interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)

                    if self.args.test_one is True:
                        out_root = os.path.join('results', self.args.arch, self.args.test_model + "_one")  # results/srcnn/crnn
                        LR_path = os.path.join(out_root,'LR')
                        SR_path = os.path.join(out_root,'SR')
                        HR_path = os.path.join(out_root,'HR')
                        os.makedirs(LR_path, exist_ok=True)
                        os.makedirs(SR_path, exist_ok=True)
                        os.makedirs(HR_path, exist_ok=True)
                        # LR
                        LR = torchvision.utils.make_grid(tensor_in, nrow=1, padding=0)
                        LR_label = str_filt(pred_str_lr[i], 'lower') + '_.png'
                        LR_label = LR_label.replace('/', '')
                        torchvision.utils.save_image(LR, os.path.join(LR_path, LR_label), padding=0)
                        # SR
                        SR = torchvision.utils.make_grid(tensor_out, nrow=1, padding=0)
                        SR_label = str_filt(pred_str_sr[i],'lower') + '_.png'
                        SR_label = SR_label.replace('/', '')
                        torchvision.utils.save_image(SR, os.path.join(SR_path, SR_label), padding=0)
                        # HR
                        HR = torchvision.utils.make_grid(tensor_target, nrow=1, padding=0)
                        HR_label = str_filt(label_strs[i], 'lower') + '_.png'
                        HR_label = HR_label.replace('/', '')
                        torchvision.utils.save_image(HR, os.path.join(HR_path, HR_label), padding=0)
                    else:
                        images = ([tensor_in, tensor_out, tensor_target])
                        vis_im = torch.stack(images)
                        vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                        out_root = os.path.join('results', self.args.arch, self.args.test_model)
                        if not os.path.exists(out_root):
                            os.makedirs(out_root, exist_ok=True)
                        im_name = str_filt(pred_str_lr[i], 'lower') + '_' + str_filt(pred_str_sr[i],'lower') + '_' + str_filt(label_strs[i], 'lower') + '_.png'
                        im_name = im_name.replace('/', '')
                        torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG_list, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, recognizer=None, prefix="acc", global_model=None):

        ckpt_path = self.ckpt_path# = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path,exist_ok=True)
        for i in range(len(netG_list)):
            netG_ = netG_list[i]

            net_state_dict = None
            if self.config.TRAIN.ngpu > 1:
                netG = netG_.module
            else:
                netG = netG_

            save_dict = {
                'state_dict_G': netG.state_dict(),
                'info': {'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                         'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
                'best_history_res': best_acc_dict,
                'best_model_info': best_model_info,
                'param_num': sum([param.nelement() for param in netG.parameters()]),
                'converge': converge_list,
            }

            if is_best:
                torch.save(save_dict, os.path.join(ckpt_path, 'model_best' + '.pth'))
            else:
                torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

        if is_best:
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        rec_state_dict = recognizer[i].state_dict()
                        # torch.save(rec_state_dict, os.path.join(ckpt_path, 'recognizer_best_' + prefix + '_' + str(i) + '.pth'))
                        torch.save(rec_state_dict, os.path.join(ckpt_path, 'recognizer_best_' + prefix + '.pth'))
                else:
                    rec_state_dict = recognizer.state_dict()
                    torch.save(rec_state_dict, os.path.join(ckpt_path, 'recognizer_best.pth'))
            if not global_model is None:
                torch.save(global_model, os.path.join(ckpt_path, 'global_model_best.pth'))
        else:
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_' + str(i) + '.pth'))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer.pth'))
            if not global_model is None:
                torch.save(global_model.state_dict(), os.path.join(ckpt_path, 'global_model.pth'))


    def parse_abinet_data(self, imgs_input):
        return imgs_input # 对输入图像做预处理。这里的abinet不需要做任何操作

    def parse_parseq_data(self, imgs_input):
        return imgs_input


    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        print("recognizer_path:", recognizer_path)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        if recognizer_path is None:
            model.load_state_dict(stat_dict)
        else:
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        return model, aster_info

    def CRNNRes18_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN_ResNet18(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        if recognizer_path is None:
            if stat_dict == model.state_dict():
                model.load_state_dict(stat_dict)
        else:
            model = stat_dict
        return model, aster_info

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):
        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor


    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
