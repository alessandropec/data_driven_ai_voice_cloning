from hello.deepWorld.utils import RunningAverage, save_checkpoint
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter


def train_epoch(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()

def evaluate_accuracy_loss(model, device, test_loader,loss_fn):
	correct = 0
	total = 0
	model.eval()
	loss_avg = RunningAverage()
	with torch.no_grad():
		with tqdm(total=len(test_loader)) as t:
			for batch_idx, data in enumerate(test_loader):
				inputs = data[0].to(device)
				target = data[1].to(device)

				outputs = model(inputs)
				loss = loss_fn(outputs, target)

				_, predicted = torch.max(outputs.data, 1)
				total += target.size(0)
				correct += (predicted == target).sum().item()

				loss_avg.update(loss.item())

				t.set_postfix({"loss":'{:05.3f}'.format(loss_avg()),
							   "acc":(100*correct/total)})
			
				t.update()

	return (100*correct/total),loss_avg()


#TO DO:
#  implementare la non sovrascittura del best modeltramite best_acc
#  capire se ha senso implementare la valutazione dell'accuracy durante il training evitando di re valutare ogni fine epoca
def train_and_evaluate_accuracy(model, device, train_loader, val_loader, 
                                optimizer, loss_fn, params, scheduler=None,last_epoch=0,best_acc=0.0,re_evaluate_train=True):
    writer = SummaryWriter(params.logdir,comment=params.dataset_name)
    for epoch in range(params.epochs):
        print("Training...")
        train_epoch(model, device, train_loader, optimizer, loss_fn)
        
        if re_evaluate_train:
            print("Re-evaluate training...")
            acc_train,avg_loss_train = evaluate_accuracy_loss(model,device,train_loader,loss_fn)

        print("Evaluate testing...")
        acc_val,avg_loss_val = evaluate_accuracy_loss(model, device, val_loader,loss_fn)
        
        print("Epoch {}/{} Train Loss:{} Val Loss: {} Train Acc:{} Valid Acc:{}".format(epoch+1+last_epoch, params.epochs+last_epoch, avg_loss_train,avg_loss_val, acc_train,acc_val))

        is_best = (acc_val > best_acc)
        if is_best:
            best_acc = acc_val
        if scheduler:
            scheduler.step()

        save_checkpoint({"epoch": epoch+last_epoch+1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, "{}".format(params.checkpoint_dir))
        writer.add_scalar("data{}/trainingLoss".format(params.dataset_name), avg_loss_train,global_step=epoch+last_epoch+1)
        writer.add_scalar("data{}/valLoss".format(params.dataset_name), avg_loss_val,global_step=epoch+last_epoch+1)
        
        writer.add_scalar("data{}/trainingAccuracy".format(params.dataset_name), acc_train, global_step=epoch+last_epoch+1)
        writer.add_scalar("data{}/valAccuracy".format(params.dataset_name), acc_val, global_step=epoch+last_epoch+1)
    writer.close()