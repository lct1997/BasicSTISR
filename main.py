import yaml
import argparse
import os
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from setup import Logger


def main(config, args):
    Mission = TextSR(config, args)
    if args.test:
        Mission.test()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--STN', action='store_true', default=True, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--mask', action='store_true', default=True, help='')
    parser.add_argument('--resume', type=str, default=None,help='resume or test file(.pth)')
    parser.add_argument('--resume_train', action='store_true', default=False, help='resume flag')
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--arch', default='lemma', choices=['srcnn', 'srres', 'tsrn', 'tbsrn', 'tg', 'tpgsr','tatt'])
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--test_one', action='store_true', default=False, help='test one by one')
    parser.add_argument('--test_model', type=str, default='CRNN',
                        choices=['ASTER', "CRNN", "MORAN", 'ABINet', 'MATRN', 'PARSeq'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--lr_position', type=float, default=1e-4, help='fine tune for position aware module')
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.Loader)
    config = EasyDict(config)
    config.TRAIN.lr = args.learning_rate
    parser_TPG = argparse.ArgumentParser()
    if args.test == True:
        Logger.init(f'experiments/{args.arch}', f'{args.arch.upper()}', 'test')
    else:
        Logger.init(f'experiments/{args.arch}', f'{args.arch.upper()}', 'train')
    Logger.enable_file()
    main(config, args)
