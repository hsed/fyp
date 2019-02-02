import torch
import torch.optim as optim
from lib.progressbar import progress_bar


def train_epoch(model, criterion, optimizer, train_loader, device=torch.device('cuda'), dtype=torch.float):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, dtype), targets.to(device, dtype)
        #print("TARGETS:", targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print("OUTPUTS:", outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        #print("LOSS:",loss.item())
        #quit()

        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader),
                     'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))    #
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))


def val_epoch(model, criterion, val_loader, device=torch.device('cuda'), dtype=torch.float):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(
                device, dtype), targets.to(device, dtype)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader),
                         'Val Loss: {0:.4e}'.format(val_loss/(batch_idx+1)))
            #print('loss: {0: .4e}'.format(val_loss/(batch_idx+1)))


def test_epoch(model, test_loader, result_collector, device=torch.device('cuda'), dtype=torch.float):
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets, extra) in enumerate(test_loader):
            ## test_loader should have a final transformer which returns
            ## `y_gt_mm_not_centered` i.e. straight from dataset
            ## `x_cropped_scaled` i.e. no aug but perform crop based on CoM (from dataset) and scale according
            ## to the model requirements i.e 128x128
            ## `CoM point` <-- this will be used to recover back y_pred_mm_not_centered
            # y_pred_std_centered # in mm but standardised
            # values are i.e. [-1, 1]
            ## need special function forward eval to now also pass through last pca layer ans get back
            ## 1, 63  instead of 1, 30
            outputs = model.forward_eval(inputs.to(device, dtype))

            # result college will convert y ->
            result_collector((inputs, outputs, targets, extra))

            progress_bar(batch_idx, len(test_loader))


def test_epoch_dropout(model, test_loader, result_collector, device=torch.device('cuda'), dtype=torch.float, output_rate=10):
	'''
		use dropouts to produce output_rate outputs for each input and then use the mean as output
		we could additionally do the above and supply multiple outcomes.

		important please use batch_size=1 for data_loader!
		
		`output_rate` => number of evaluations or outputs per input sample
	'''
	## special case here we will be doing dropout and using the mean value ideally...
	## in this case all dropouts are used, maybe in future we would want to use only one dropout
	model.train()

	with torch.no_grad():
		for batch_idx, (inputs, targets, extra) in enumerate(test_loader):
			assert(targets.shape[0] == 1) # only works currently for batch_size=1
			output_set = torch.cat([model.forward_eval(inputs.to(device, dtype)) for _ in range(output_rate)])
			
			#print("Set shape: ", output_set.shape)
			
			## do torch.mean in the right axis
			output_mean = torch.mean(output_set, dim=0, keepdim=True)

			#print("Mean Shape: ", output_mean.shape)
			#quit()
			
			# result college will convert y -> 
			result_collector((inputs, output_mean, targets, extra))

			progress_bar(batch_idx, len(test_loader))
