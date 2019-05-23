import os
import argparse
import torch
from tqdm import tqdm

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
    config['data_loader']['validation_split'] = -1.0 # ensure entire test set is used

    train_data_loader = get_instance(module_data, 'data_loader', config)
    data_loader = train_data_loader.split_validation()

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    #model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, config['dtype'])
    target_dtype = getattr(torch, config['target_dtype'])

    #### important function to instantiate Avg3DError properly
    #### train data loader is only used to get some parameters
    init_metrics(metric_fns, model, train_data_loader, device, dtype)

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    
    
    # because config is loaded from test, model remains in 'train_mode'
    # output is low-dim output
    model = model.to(device, dtype)
    model.eval()

    # if getattr(model, 'train_mode', False):
    #     model.train_mode = False # to perform De-PCA

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, target) in enumerate(data_loader):

                target = _tensor_to(target, device, dtype) # no more target_dtype not required all types must be same
                data = _tensor_to(data, device, dtype)
                output = model(data)
                
                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data_loader.batch_size
                
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    metric =  metric(output, target) * batch_size
                    pbar.set_description('Metric: %0.4f' % (metric/batch_size))
                    total_metrics[i] += metric#metric(output, target) * batch_size
                pbar.update(1)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='index of GPU to enable')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    if args.resume:
        config = torch.load(args.resume)['config']

    main(config, args.resume)
