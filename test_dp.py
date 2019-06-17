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
    data_loader = module_data.DeepPriorTestDataLoader( #getattr(module_data, config['data_loader']['type'])
        config['data_loader']['args']['data_dir'],
        batch_size=2048,#4,
        shuffle=False,
        validation_split=0.0,
        dataset_type='test',
        num_workers=config['data_loader']['args']['num_workers'],
        ### custom enable for hpe, disable for har
        test_mm_err=True,
        use_msra=config['data_loader']['args']['use_msra']
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    #model.summary()
    #model.initialize_weights(data_loader.pca_weights_np, data_loader.pca_bias_np)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = []#[module_metric.Avg3DError]#[getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, config['dtype'])

    # if getattr(model, 'final_layer.initialize_weights', False):
    print("Initialising PCA weights")
    model.final_layer.initialize_weights(data_loader.pca_weights_np, data_loader.pca_bias_np)

    # because config is loaded from test, model remains in 'train_mode'
    # output is low-dim output
    model = model.to(device, dtype)
    model.eval()
    
    if getattr(model, 'train_mode', False):
        model.train_mode = False # to perform De-PCA

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, target, extra) in enumerate(data_loader):

                
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
                # print("COM: ", extra[0][2])
                # print("Y_TEST:\n", target[0][:10])
                # print('OUTPUT:\n', output[0][:10])
                # print("ITEM_X:\n", data[0][0, 64,60:90])
                # quit()
                target = target.to(device, dtype)
                
                # computing loss, metrics on test set
                #loss = loss_fn(output, target) ## temporary disabled
                batch_size = data_loader.batch_size
                
                #total_loss += loss.item() * batch_size
                # for i, metric in enumerate(metric_fns):
                #     metric =  metric(output, target) * batch_size
                #     pbar.set_description('Metric: %0.4f' % (metric/batch_size))
                #     total_metrics[i] += metric#metric(output, target) * batch_size
                data_loader.test_res_collector((data, output, target, extra))

                pbar.set_description("[AVG_3D_ERR: %0.4fmm]" % data_loader.test_res_collector.calc_avg_3D_error())
                pbar.update(1)

                #break

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)

    print("\nFINAL_AVG_3D_ERROR: %0.4fmm\n" % data_loader.test_res_collector.calc_avg_3D_error())


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
