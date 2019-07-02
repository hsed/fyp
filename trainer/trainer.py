import numpy as np
import torch
from torchvision.utils import make_grid
from trainer import BaseTrainer, ensure_dir
from data_utils import PersistentDataLoader
from metrics import Avg3DError
from models import PCADecoderBlock
from tqdm import tqdm
from data_utils import plotImg

# to check instances for sample plotting
from models import DeepPriorPPModel, CombinedModel

def init_metrics(metrics: list, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                device, dtype):
    '''
        A standard helper module to help instantiate metrics that are class-based
        
        Currently only Avg3DError metric object is supported
    '''
    if Avg3DError in metrics:
            '''
                This special metric requires a PCA decoder class with correct
                parameters i.e. weight and bias matx pre-learnt during PCA training.

                Currently, the implementation is such that the dataloader class for HPE
                loads PCA and saves weights and biases, now these are automatically
                initialised and can be accessed from the dataloader class.

                Note: PCA is always learnt on training data with likely data augmentation
                We replace the uninitialised reference with the initialised one here.

            '''
            #print("[INIT_METRICS] Initialising Avg3DError Metric with Params: ", data_loader.params)

            idx = metrics.index(Avg3DError)
            ### init metric classes for future use
            
            #### new updated method
            #### make this a global fn
            #from inspect import getargspec

            # if init_pca_layer is found then assume model always outputs non_pca_layer by default
            # if eval_pca_space or train_pca_space is found then if True only then pca output is used
            
            eval_pca_space = True # this means that the model will output Nx30 tensor during training
            train_pca_space = True # this means that the model will output Nx30 tensor during testing
            
            if getattr(model, 'initialize_pca_layer', False):
                print("[INIT_METRICS] Attempting to initialize PCA layer in model...")
                model.initialize_pca_layer(w=data_loader.pca_weights_np, b=data_loader.pca_bias_np)
                
                # If not found -- assume these are are false because assume the model will
                # make use of pca layer
                eval_pca_space = getattr(model, 'eval_pca_space', False)
                train_pca_space = getattr(model, 'train_pca_space', False)
            else:
                print('[INIT_METRICS] No PCA Layer found in model, no PCA initialization attempted.')
                
                # assume model outputs in pca_space as there is no decoder to project back to 3D space
                eval_pca_space = True
                train_pca_space = True

            print('[INIT_METRICS] AVG3DError expects %s during training and %s during testing' % \
                 ('PCA_30_DIM' if train_pca_space else 'JOINTS_63', 'PCA_30' if eval_pca_space else 'JOINTS_63'))            
            avg_3d_err_metric = Avg3DError(cube_side_mm=data_loader.params['cube_side_mm'],
                                           ret_avg_err_per_joint=False,
                                           train_pca_space=train_pca_space,
                                           eval_pca_space=eval_pca_space)
            
            # do this always because pca_decoder will still be needed during training
            avg_3d_err_metric.init_pca(dict(num_joints=data_loader.params['num_joints'],
                                        num_dims=data_loader.params['world_dim'],
                                        pca_components=data_loader.params['pca_components']),
                                        weight_np=data_loader.pca_weights_np,
                                        bias_np=data_loader.pca_bias_np,
                                        device=device, dtype=dtype)

            # nothing is returned but the metrics list is modified
            metrics[idx] = avg_3d_err_metric

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        # persistent wrapper to load all data to RAM
        self.data_loader = PersistentDataLoader(data_loader, load_on_init=True) \
                                                if config['trainer']['persistent_storage'] else data_loader
        self.valid_data_loader = PersistentDataLoader(valid_data_loader, load_on_init=True) \
                                                if config['trainer']['persistent_storage'] else valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1#int(np.sqrt(data_loader.batch_size))

        if self.no_train:
            print("[TRAIN] TRAIN DISABLED: NO MONITOR NO LOGS NO SAVE")
            self.epochs = 1
            self.start_epoch = 1


        ## special case for HPE

        init_metrics(self.metrics, self.model, self.data_loader, self.device, self.dtype) # todo
        

        if self.only_save:
            # mutually exclusive to resume
            print('[TRAINER] Training disabled, only performing single validation, also disabling monitor')
            ### make sure to delete this only save option from config!!
            ### this key will always be present because only then this if condition will be true!
            self.config['trainer']['only_save'] = False
            ensure_dir(self.checkpoint_dir) # call this to ensure directories are created accordingly
            self._save_config()
            self._save_checkpoint(1, save_best=True)
            quit()

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        print("\n") # clearer

        self.writer.set_step(epoch)
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        #tqdm_dataloader = tqdm(self.data_loader)
        with tqdm(total=len(self.data_loader)) as tqdm_pbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                
                target = self._tensor_to(target)
                data = self._tensor_to(data)
                output = self.model(data)

                if batch_idx == 0:
                    # print histogram of first batch
                    self.writer.add_histogram('y_hist_b0', output[0].data.detach().cpu())
                    # print histogram of model
                    with torch.no_grad():
                        for name,param in self.model.named_parameters():
                            if 'hpe.main_layers' in name:
                                self.writer.add_histogram('%s_hist_b0' % name, param.detach().cpu())
                                break # break for loop
                #print(data[0].shape)
                #print(data[0])
                #quit()
                # if isinstance(data, torch.Tensor):
                #     data = data.to(self.device, self.dtype)
                #     output = self.model(data)
                # elif isinstance(data, tuple):
                #     # if its not a tensor its probably a tuple
                #     # we expect model to handle tuple
                #     # we send it in similar fashion to *args
                #     data = tuple(sub_data.to(self.device, self.dtype) for sub_data in data)
                #     output = self.model(*data)
                # else:
                #     raise RuntimeError("Invalid Datatype")
                
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                ## do this every epoch not every batch
                #self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
                #self.writer.add_scalar('loss', loss.item())
                batch_metrics = self._eval_metrics(output, target)

                total_loss += loss.item()
                total_metrics += batch_metrics

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    # self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    #     epoch,
                    #     batch_idx * self.data_loader.batch_size,
                    #     self.data_loader.n_samples,
                    #     100.0 * batch_idx / len(self.data_loader),
                    #     loss.item()))
                    #({:.0f}%)
                    tqdm_pbar.update(self.log_step)
                    tqdm_pbar.set_description('Train [Epoch: {}/{}] [Metrics: {}] [Loss: {:.4f}]'.format(
                        epoch,
                        self.epochs,
                        #(batch_idx+1) * self.data_loader.batch_size,
                        #self.data_loader.n_samples, ##100.0 * batch_idx / len(self.data_loader),
                        np.array2string(batch_metrics, precision=2, formatter={'float_kind':lambda x: "%0.2f" % x}),
                        loss.item())
                    )
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                #if self.data_loader.reduce: break # return after one batch
                #break

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        ## per epoch logging 
        self.writer.set_step(epoch)
        self.writer.add_scalar('loss', log['loss'])

        for metric, metric_val in zip(self.metrics, log['metrics']):
            self.writer.add_scalar(f'{metric.__name__}', metric_val)
        

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        # the no_grad context will ensure that avg3Derror recognizes it is in valid mode
        with torch.no_grad():
            for batch_idx, (data, target) in \
                tqdm(enumerate(self.valid_data_loader), desc='Validate', total=len(self.valid_data_loader)):
                
                # supports tuples and tensors
                target = self._tensor_to(target)

                self.optimizer.zero_grad()

                data = self._tensor_to(data)
                
                # no more tuple unwrapping, all unwrapping if required is done by loss fn
                output = self.model(data)

                # if isinstance(data, torch.Tensor):
                #     data = data.to(self.device, self.dtype)
                #     output = self.model(data)
                # elif isinstance(data, tuple):
                #     # if its not a tensor its probably a tuple
                #     # we expect model to handle tuple
                #     # we send it in similar fashion to *args
                #     data = tuple(sub_data.to(self.device, self.dtype) for sub_data in data)
                #     output = self.model(*data)
                # else:
                #     raise RuntimeError("Invalid Datatype")

                # support for tuple type targets and outputs is implemented in the loss function itself.
                loss = self.loss(output, target)

                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                #self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                #if self.data_loader.debug: break #for debugging

        log = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
        
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', log['val_loss'])

        for metric, metric_val in zip(self.metrics, log['val_metrics']):
            self.writer.add_scalar(f'{metric.__name__}', metric_val)

        return log


    def _tensor_to(self, data):
        ## custom function
        ## note packed sequence is also of type tuple so its a special case which raises errors
        ## code here is an attempt to fix it
        if isinstance(data, torch.Tensor) or isinstance(data, torch.nn.utils.rnn.PackedSequence):
            data = data.to(self.device, self.dtype)
            return data
        elif isinstance(data, tuple):
            # if its not a tensor its probably a tuple
            # we expect model to handle tuple
            # we send it in similar fashion to *args
            data = tuple(sub_data.to(self.device, self.dtype) for sub_data in data)
            return data
        else:
            raise RuntimeError("Invalid Datatype %s" % type(data))
    

    def _predict_and_write_2D(self, sample_data , epoch , mm2pxfn):
        '''
            (sample[DT.DEPTH_ORIG], sample[DT.DEPTH_CROP], sample[DT.JOINTS_ORIG_PX], \
            sample[DT.COM_ORIG_PX], sample[DT.CROP_TRANSF_MATX], \
            None, sample[DT.DEPTH_CROP_AUG], sample[DT.AUG_TRANSF_MATX], \
            sample[DT.AUG_MODE], sample[DT.AUG_PARAMS], sample[DT.COM_ORIG_MM], sample[DT.DEPTH], sample[DT.JOINTS])

            This function prints out a sample prediction to tensorboard in the form of add figure
            It requires some sample data, an mm2pxfn and the fact that Avg3DError object is present in
            the currently used metrics
        '''
        #return
        #print("WE PRED AND WRITE", self.metrics)
        m_id = -1
        for i, metric in enumerate(self.metrics):
            if isinstance(metric, Avg3DError):
                m_id = i
                break
        from models import DeepPriorPPModel
        if m_id > -1 and isinstance(self.model,DeepPriorPPModel):
            #print("SAMPLE DATA LEN:", len(sample_data))
            #quit()
            ## code req -- confirm its the hpe we are training 
            # now only works if input is of tuple type
            ## need to document this function!
            ## if no action is used then action item is just none...
            with torch.no_grad():
                x,y = self._tensor_to((torch.from_numpy(sample_data[-2][0]), torch.tensor([sample_data[-2][1]]))), \
                        self._tensor_to(torch.from_numpy(sample_data[-1]))
                x0,y = x[0].unsqueeze(0), y.unsqueeze(0) # input, target
                y_ = self.model((x0, x[1])) # output
                
                #print("y_", y_.shape)
                # all 3 are torch tensors!
                error_3d, y_mm_, y_mm = self.metrics[m_id](y_, y, return_mm_data=True)

                # all are tensors return 0d, 1,21,3; 1,21,3
                #print("types: ", type(error_3d), type(y_mm_), type(y_mm), "shapes: ", error_3d.shape, y_mm_.shape, y_mm.shape)
                
                com_orig = sample_data[-3]
                #print("PLOT DEBUG FILENAME: ", sample_data[-4])
                #print("com_Shape", com_orig.shape)
                

                output_mm_uncentered = y_mm_.squeeze(0).detach().cpu().numpy() + com_orig
                #print("out_mm_uncent", output_mm_uncentered.shape)
                error_3d = error_3d.item() # extract value from 0d tensor

                out_px = mm2pxfn(output_mm_uncentered)


                fig = plotImg(*sample_data[:5], show_aug_plots=False, return_fig=True, keypt_pred_px=out_px, pred_err=error_3d)
                #from matplotlib import pyplot as plt
                #plt.figure(fig.number)
                
                #plt.show()
                
                self.writer.add_figure('sample_prediction', fig, epoch)

            #exit()
        else:
            pass #print("AVG3DERROR NOT FOUND")
