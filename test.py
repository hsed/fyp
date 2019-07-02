import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import yaml

import models as module_arch
from models import PCADecoderBlock
from data_utils import data_loaders as module_data
from metrics import loss as module_loss
from metrics import metric as module_metric
from metrics import Avg3DError
from train import get_instance
from trainer import init_metrics

def _tensor_to(data, device, dtype):
        ## custom function
        if isinstance(data, torch.Tensor) or isinstance(data, torch.nn.utils.rnn.PackedSequence):
            data = data.to(device, dtype)
            return data
        elif isinstance(data, tuple):
            # if its not a tensor its probably a tuple
            # we expect model to handle tuple
            # we send it in similar fashion to *args
            data = tuple(sub_data.to(device, dtype) for sub_data in data)
            return data
        else:
            raise RuntimeError("Invalid Datatype %s" % type(data))

def main(config, resume):
    # setup data_loader instances
    #config['data_loader']['validation_split'] = -1.0 # ensure entire test set is used
    train_data_loader = get_instance(module_data, 'data_loader', config)
    data_loader = train_data_loader.split_validation()

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    #model.summary()

    if isinstance(config['loss'], str):
        # do the normal thing, legacy
        loss_fn = getattr(module_loss, config['loss'])
    else:
        # assume loss is of dict type with same format as other classes
        # the loss instance should have __call__ implemented
        loss_fn = get_instance(module_loss, 'loss', config)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, config['dtype'])

    # load state dict
    print("[TEST] Loading state_dict from checkpoint: %s" % resume)
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # ### custom code
    # if new_attn is not None:
    #     del model.attention_layer

    #### important function to instantiate Avg3DError properly
    #### train data loader is only used to get some parameters
    #### if you run this function AFTER state_dict restore then pca weights
    #### for model won't be loaded but note that there can be possible mismatches
    #### still if pca weights in npz file doesn't match the model
    #### so if you do this for every new pca you would have to retrain the model or
    #### simply load and then do save_only.
    init_metrics(metric_fns, model, train_data_loader, device, dtype)
    
    # because config is loaded from test, model remains in 'train_mode'
    # output is low-dim output
    model = model.to(device, dtype)
    model.eval()

    # if getattr(model, 'train_mode', False):
    #     model.train_mode = False # to perform De-PCA

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    batch_metrics = np.zeros(len(metric_fns))

    #last_batch_size = len(data_loader)*data_loader.batch_size - len(data_loader.dataset)
    n_samples = len(data_loader.sampler)
    samples_left = n_samples #len(data_loader.dataset) # as if batch_size == 1

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, target) in enumerate(data_loader):

                target = _tensor_to(target, device, dtype) # no more target_dtype not required all types must be same
                data = _tensor_to(data, device, dtype)
                output = model(data)
                
                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                if samples_left < data_loader.batch_size:
                    batch_size = samples_left    
                else:
                    batch_size = data_loader.batch_size
                #print("batch_sz", batch_size)
                samples_left -= data_loader.batch_size
                
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    metric =  metric(output, target) * batch_size
                    #pbar.set_description('Metric: %0.4f' % (metric/batch_size))
                    batch_metrics[i] = (metric/batch_size)
                    total_metrics[i] += metric#metric(output, target) * batch_size
                
                pbar.set_description('Metric: %s' % np.array2string(batch_metrics, formatter={'float_kind':lambda x: "%0.2f" % x}))
                pbar.update(1)

    
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', required=True, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='index of GPU to enable')
    parser.add_argument('-fa', '--fusion_alpha', type=float, default=None) #temp
    parser.add_argument('-ae', '--action-equiprob', default=None, type=float,
                        help='Action Equiprob Condition')
    parser.add_argument('-bs', '--batch_size', default=None, type=int,
                        help='overwrite batch_size used in config')
    parser.add_argument('-vs', '--val-split', default=None, type=float,
                        help='-1.0 < split < 0.0 ')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.fusion_alpha is not None:
        print("USING HPE FUSION ALPHA: ", args.fusion_alpha)
        config['arch']['args']['act_0c_alpha'] = args.fusion_alpha
    if args.action_equiprob is not None:
        if config['arch']['type'] == 'CombinedModel':
            print('OVERWRITING COMBINED_HPE_ACT ACTION_EQUIPROB TO %f' % args.action_equiprob)
            config['arch']['args']['hpe_args']['action_equiprob_chance'] = args.action_equiprob
        else:
            print('OVERWRITING HPE ACTION_EQUIPROB TO %f' % args.action_equiprob)
            config['arch']['args']['action_equiprob_chance'] = args.action_equiprob
        #config['data_loader']['args']['validation_split'] = -0.2
    # if args.attention_args:
    #     print("USING CUSTOM ACTION ARGS! Old:", config['arch']['args']['attention_type'], "New", args.action_args)
    #     config['arch']['args']['attention_type_new'] = args.attention_args
    if args.batch_size is not None:
        print('OVERWRITING BATCH_SIZE TO %d' % args.batch_size)
        config['data_loader']['args']['batch_size'] = args.batch_size
    if args.val_split is not None:
        config['data_loader']['args']['validation_split'] = args.val_split
    #config['arch']['args']['use_unrolled_lstm'] = False #True
    #config['data_loader']['args']['pad_sequence'] = True
    #print(yaml.dump(config))
    main(config, args.resume)
