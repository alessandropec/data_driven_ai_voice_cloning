import torch
import os,shutil,json
from tqdm import tqdm

class Params():
	def __init__(self, json_path):
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)
	def save(self, json_path):
		with open(json_path, 'w') as f:
			params = json.dump(self.__dict__, f, indent=4)
	def update(self, json_path):
		with open(json_path) as f:
			params = json.load(f)
			self.__dict__.update(params)

class RunningAverage():
	def __init__(self):
		self.total = 0
		self.steps = 0
	def update(self, loss):
		self.total += loss
		self.steps += 1
	def __call__(self):
		return (self.total/float(self.steps))

def evaluate(model, device, test_loader,loss_fn):
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

def get_best_device():
	return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, checkpoint):
	filename = os.path.join(checkpoint, 'last.pth.tar')
	if not os.path.exists(checkpoint):
		print(f"Checkpoint directory {checkpoint} does not exist, i creating it for you...")
		os.mkdir(checkpoint)
	torch.save(state, filename)
	if is_best:
		print(f"I have found the best model ever, i saving it for you...")
		shutil.copyfile(filename, os.path.join(checkpoint, "model_best.pth.tar"))