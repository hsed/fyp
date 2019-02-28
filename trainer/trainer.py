import numpy as np
import torch
from torchvision.utils import make_grid
from trainer import BaseTrainer
from data_utils import PersistentDataLoader

from tqdm import tqdm

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
                                                if config['trainer']['persistent_storage'] else data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1#int(np.sqrt(data_loader.batch_size))

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
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        #tqdm_dataloader = tqdm(self.data_loader)
        with tqdm(total=len(self.data_loader)) as tqdm_pbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                
                self.optimizer.zero_grad()

                target = target.to(self.device, self.target_dtype)
                if isinstance(data, torch.Tensor):
                    data = data.to(self.device, self.dtype)
                    output = self.model(data)
                elif isinstance(data, tuple):
                    # if its not a tensor its probably a tuple
                    # we expect model to handle tuple
                    # we send it in similar fashion to *args
                    data = tuple(sub_data.to(self.device, self.dtype) for sub_data in data)
                    output = self.model(*data)
                else:
                    raise RuntimeError("Invalid Datatype")
                
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                ## do this every epoch not every batch
                #self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
                #self.writer.add_scalar('loss', loss.item())
                
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target)

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    # self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    #     epoch,
                    #     batch_idx * self.data_loader.batch_size,
                    #     self.data_loader.n_samples,
                    #     100.0 * batch_idx / len(self.data_loader),
                    #     loss.item()))
                    #({:.0f}%)
                    tqdm_pbar.update(self.log_step)
                    tqdm_pbar.set_description('Train Epoch: {} [{}/{}] Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        #100.0 * batch_idx / len(self.data_loader),
                        loss.item()))
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                #break # return after one batch
        

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        ## per epoch logging 
        self.writer.set_step(epoch)
        self.writer.add_scalar('loss', log['loss'])

        for metric, metric_val in zip(self.metrics, log['metrics']):
            self.writer.add_scalar(f'{metric.__name__}', metric_val)
        

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                target = target.to(self.device, self.target_dtype)

                self.optimizer.zero_grad()

                if isinstance(data, torch.Tensor):
                    data = data.to(self.device, self.dtype)
                    output = self.model(data)
                elif isinstance(data, tuple):
                    # if its not a tensor its probably a tuple
                    # we expect model to handle tuple
                    # we send it in similar fashion to *args
                    data = tuple(sub_data.to(self.device, self.dtype) for sub_data in data)
                    output = self.model(*data)
                else:
                    raise RuntimeError("Invalid Datatype")

                loss = self.loss(output, target)

                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                #self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
        
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', log['val_loss'])

        for metric, metric_val in zip(self.metrics, log['val_metrics']):
            self.writer.add_scalar(f'{metric.__name__}', metric_val)

        return log
