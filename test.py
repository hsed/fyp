import os
import argparse
import torch
from tqdm import tqdm

import models as module_arch
from data_utils import data_loaders as module_data
from metrics import loss as module_loss
from  metrics import metric as module_metric
from train import get_instance


def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=4,
        shuffle=False,
        validation_split=0.0,
        dataset_type='test',
        num_workers=2
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, config['dtype'])
    target_dtype = getattr(torch, config['target_dtype'])
    model = model.to(device, dtype)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):

            target = target.to(device, target_dtype)
            if isinstance(data, torch.Tensor):
                data = data.to(device, dtype)
                output = model(data)
            elif isinstance(data, tuple):
                # if its not a tensor its probably a tuple
                # we expect model to handle tuple
                # we send it in similar fashion to *args
                data = tuple(sub_data.to(device, dtype) for sub_data in data)
                output = model(*data)
            else:
                raise RuntimeError("Invalid Datatype")
            #
            # save sample images, or do something with output here
            #
            
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data_loader.batch_size
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
