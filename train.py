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
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    print("[NEW] METRICS: ", metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    #lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer) TODO: Enable later..

    print("\n=> Building trainer...")
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=None,#lr_scheduler,
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
    parser.add_argument('-da', '--data-aug', default=None, nargs='+', type=int,
                        help='[0,1,2,3]')
    parser.add_argument('-pda', '--pca-data-aug', default=None, nargs='+', type=int,
                        help='[0,1,2,3]')
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
    
    if args.data_aug is not None:
        config['data_loader']['args']['data_aug'] = args.data_aug
    if args.pca_data_aug is not None:
        config['data_loader']['args']['pca_data_aug'] = args.pca_data_aug

    if args.no_log:
        print("[MAIN] Logging + saving disabled due to cmd flag!")
        del config['trainer']['monitor'] # disable monitoring
        config['trainer']['tensorboardX'] = False # disable logging

    main(config, args.resume)
