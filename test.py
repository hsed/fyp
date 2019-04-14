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
        batch_size=2048,#4,
        shuffle=False,
        validation_split=0.0,
        dataset_type='test',
        num_workers=config['data_loader']['args']['num_workers'],
        ### custom enable for hpe, disable for har
        test_mm_err=False,#True,
        use_msra=False
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    #model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [module_metric.Avg3DError]#[getattr(module_metric, met) for met in config['metrics']]

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

    ### note this only needs to be done here but not for training as the trainer
    ### class actually calls the same lines of code
    if module_metric.Avg3DError in metric_fns:
        idx = metric_fns.index(module_metric.Avg3DError)
        '''
            This special metric requires a PCA decoder class with correct
            parameters i.e. weight and bias matx pre-learnt during PCA training.

            Currently, the implementation is such that the dataloader class for HPE
            loads PCA and saves weights and biases, now these are automatically
            initialised and can be accessed from the dataloader class.

            Note: PCA is always learnt on training data with likely data augmentation

            We replace the uninitialised reference with the initialised one here.

        '''
        ### init metric classes for future use
        avg_3d_err_metric = module_metric.Avg3DError(cube_side_mm=data_loader.params['cube_side_mm'],
                                                    ret_avg_err_per_joint=False)
        
        avg_3d_err_metric.pca_decoder = \
            module_arch.PCADecoderBlock(num_joints=data_loader.params['num_joints'],
                            num_dims=data_loader.params['world_dim'],
                            pca_components=data_loader.params['pca_components'])
        
        ## weights are init as transposed of given
        avg_3d_err_metric.pca_decoder.initialize_weights(weight_np=data_loader.pca_weights_np,
                                                            bias_np=data_loader.pca_bias_np)
        avg_3d_err_metric.pca_decoder= avg_3d_err_metric.pca_decoder.to(device, dtype)
        
        metric_fns[idx] = avg_3d_err_metric
    
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
                #loss = loss_fn(output, target) ## temporary disabled
                batch_size = data_loader.batch_size
                
                #total_loss += loss.item() * batch_size
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

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
