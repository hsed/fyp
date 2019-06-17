import os
import yaml
import argparse
import torch
import numpy as np
import data_utils.data_loaders as module_data
import metrics.loss as module_loss
import metrics.metric as module_metric
import models as module_arch
from trainer import Trainer
from utils import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    train_logger = Logger()
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    print("=> Configuring data_loader...")
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    ### new, if val_split < 0.0 then use test_set for validation so
    ### that we can still do early stopping
    # if config['data_loader']['args']['validation_split'] < 0.0:
    #     ## invode data_loader class object but now it returns a rest dataset
    #     print("Info: using test set for validation as val_split was < 0")
    #     valid_data_loader = getattr(module_data, config['data_loader']['type'])(
    #                                 config['data_loader']['args']['data_dir'],
    #                                 batch_size=512, # temp out of mem errors #1024,#4,
    #                                 shuffle=False,
    #                                 validation_split=0.0,
    #                                 dataset_type='test',
    #                                 num_workers=config['data_loader']['args']['num_workers'],
    #                                 use_msra=config['data_loader']['args']['use_msra']
    #     )

    # build model architecture
    print("\n=> Building model...")
    model = get_instance(module_arch, 'arch', config)
    print("Trainable Params:", model.param_count())
    
    # get function handles of loss and metrics
    print("\n=> Building loss and optimizer modules...")
    if isinstance(config['loss'], str):
        # do the normal thing, legacy
        loss = getattr(module_loss, config['loss'])
    else:
        # assume loss is of dict type with same format as other classes
        # the loss instance should have __call__ implemented
        loss = get_instance(module_loss, 'loss', config)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    print("LOSS & METRICS: ", loss, metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params) \
                                        if (not config['trainer'].get('only_save', False) \
                                            and not config['trainer'].get('no_train', False)) \
                                        else torch.optim.Adam([torch.tensor([1,2,3])])
    
    if 'lr_scheduler' in config and config['lr_scheduler'] is not None:
        print('[TRAIN] Will be using LR Scheduler')
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer) #TODO: Enable later..
    else:
        lr_scheduler = None

    print("\n=> Building trainer...")
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,#lr_scheduler,
                      train_logger=train_logger,
                      )

    print("\nTraining...")
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-fp', '--force-pca', default=None, action='store_true',
                           help='Force re-calc of PCA cache')
    # parser.add_argument('-da', '--data-aug', default=None, nargs='+', type=int,
    #                     help='[0,1,2,3]')
    # parser.add_argument('-pda', '--pca-data-aug', default=None, nargs='+', type=int,
    #                     help='[0,1,2,3]')
    parser.add_argument('-bs', '--batch_size', default=None, type=int,
                        help='overwrite batch_size used in config')
    parser.add_argument('-vs', '--val-split', default=None, type=float,
                        help='-1.0 < split < 0.0 ')
    parser.add_argument('-lr', '--learn-rate', default=None, type=float,
                        help='LR')
    parser.add_argument('-wd', '--weight-decay', default=None, type=float,
                        help='Weight Decay')
    parser.add_argument('-cla', '--combined-loss-alpha', default=None, type=float,
                        help='Combined Loss Alpha')
    parser.add_argument('-clb', '--combined-loss-beta', default=None, type=float,
                        help='Combined Loss Beta')
    parser.add_argument('-clg', '--combined-loss-gamma', default=None, type=float,
                        help='Combined Loss Gamma')
    parser.add_argument('-ac', '--action-cond', default=None, type=float,
                        help='Action Condition only for HPE')
    parser.add_argument('-ae', '--action-equiprob', default=None, type=float,
                        help='Action Equiprob Condition only for HPE')
    parser.add_argument('-fa', '--hpe-fusion-alpha', default=None, type=float,
                        help='Action 0c Alpha for fusing two HPE results')
    parser.add_argument('-ee', '--ensemble-eta', default=None, type=float,
                        help='Ensemble Eta for fusing two HPE results')
    parser.add_argument('-ez', '--ensemble-zeta', default=None, type=float,
                        help='Ensemble Zeta for fusing two HAR results')
    parser.add_argument('-cv', '--combined-version', default=None, type=str,
                        help='Combined version as string')
    parser.add_argument('-ep', '--epochs', default=None, type=int,
                        help='num epochs')
    parser.add_argument('-nl', '--no-log', action='store_true',
                        help='turn off data logging and monitoring')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.yaml', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ## add cmd args to config, overwrite if necessary
    if args.force_pca is not None:
        config['data_loader']['args']['pca_overwrite_cache'] = args.force_pca
    
    # if args.data_aug is not None:
    #     config['data_loader']['args']['data_aug'] = args.data_aug
    # if args.pca_data_aug is not None:
    #     config['data_loader']['args']['pca_data_aug'] = args.pca_data_aug
    if args.val_split is not None:
        print('OVERWRITING VAL_SPLIT TO %f' % args.val_split)
        config['data_loader']['args']['validation_split'] = args.val_split
    if args.weight_decay is not None:
        print('OVERWRITING WEIGHT_DECAY TO %f' % args.weight_decay)
        config['optimizer']['args']['weight_decay'] = args.weight_decay
    if args.learn_rate is not None:
        print('OVERWRITING LEARN_RATE TO %f' % args.learn_rate)
        config['optimizer']['args']['lr'] = args.learn_rate
    if args.combined_loss_alpha is not None:
        print('OVERWRITING COMBINED_LOSS_ALPHA TO %f' % args.combined_loss_alpha)
        config['loss']['args']['alpha'] = args.combined_loss_alpha
    if args.combined_loss_beta is not None:
        print('OVERWRITING COMBINED_LOSS_BETA TO %f' % args.combined_loss_beta)
        config['loss']['args']['beta'] = args.combined_loss_beta
    if args.combined_loss_gamma is not None:
        print('OVERWRITING COMBINED_LOSS_GAMMA TO %f' % args.combined_loss_gamma)
        config['loss']['args']['gamma'] = args.combined_loss_gamma
    if args.batch_size:
        print('OVERWRITING BATCH_SIZE TO %d' % args.batch_size)
        config['data_loader']['args']['batch_size'] = args.batch_size
    if args.action_cond is not None:
        print('OVERWRITING HPE ACTION_COND TO %f' % args.action_cond)
        config['arch']['args']['action_cond_ver'] = args.action_cond
    if args.ensemble_eta is not None:
        print('OVERWRITING ensemble_eta TO %f' % args.ensemble_eta)
        config['arch']['args']['ensemble_eta'] = args.ensemble_eta
    if args.ensemble_zeta is not None:
        print('OVERWRITING ensemble_zeta TO %f' % args.ensemble_zeta)
        config['arch']['args']['ensemble_zeta'] = args.ensemble_zeta
    if args.combined_version is not None:
        print('OVERWRITING COMBINED_VERSION TO %s' % args.combined_version)
        config['arch']['args']['combined_version'] = args.combined_version
    if args.action_equiprob is not None:
        if config['arch']['type'] == 'CombinedModel':
            print('OVERWRITING COMBINED_HPE_ACT ACTION_EQUIPROB TO %f' % args.action_equiprob)
            config['arch']['args']['hpe_args']['action_equiprob_chance'] = args.action_equiprob
        else:
            print('OVERWRITING HPE ACTION_EQUIPROB TO %f' % args.action_equiprob)
            config['arch']['args']['action_equiprob_chance'] = args.action_equiprob
    if args.hpe_fusion_alpha is not None:
        print('OVERWRITING TWO HPE FUSION ALPHA (Act0c) TO %f' % args.hpe_fusion_alpha)
        config['arch']['args']['act_0c_alpha'] = args.hpe_fusion_alpha
    if args.epochs is not None:
        config['trainer']['epochs'] = args.epochs

    if args.no_log:
        print("[MAIN] Logging + saving disabled due to cmd flag")
        if 'monitor' in config['trainer']:
            del config['trainer']['monitor'] # disable monitoring
        config['trainer']['tensorboardX'] = False # disable logging

    main(config, args.resume)
