import os
import json
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
    if config['data_loader']['args']['validation_split'] < 0.0:
        print("Info: using test set for validation as val_split was < 0")
        valid_data_loader = getattr(module_data, config['data_loader']['type'])(
                                    config['data_loader']['args']['data_dir'],
                                    batch_size=4,
                                    shuffle=False,
                                    validation_split=0.0,
                                    dataset_type='test',
                                    num_workers=config['data_loader']['args']['num_workers']
        )

    # build model architecture
    print("\n=> Building model...")
    model = get_instance(module_arch, 'arch', config)
    print(model)
    
    # get function handles of loss and metrics
    print("\n=> Building loss and optimizer modules...")
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    print("\n=> Building trainer...")
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
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
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
