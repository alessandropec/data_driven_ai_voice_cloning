
import torch.nn as nn
from torchvision import transforms
import torch

RESNET_VERSIONS={"resnet18":{'fc_input': 512},
				 "resnet34":{'fc_input': 512},
				 "resnet50":{'fc_input': 2048},
				 "resnet101":{'fc_input': 2048},
				 "resnet152":{'fc_input': 2048}
				}


class SpResNet(nn.Module):
	def __init__(self,num_classes, pretrained=True,resnet_version="resnet18"):
		super(SpResNet, self).__init__()

		#self.pretrained=pretrained
		#self.resnet_version=resnet_version
		#self.fc_input=RESNET_VERSIONS[resnet_version].fc_input
		self.num_classes=num_classes

		self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
		self.model.fc = nn.Linear(512, self.num_classes)
		
	def forward(self, x):
		output = self.model(x)
		return output

def resNet_prepro(input):
	input_image,label=input
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	return preprocess(input_image).float(),label