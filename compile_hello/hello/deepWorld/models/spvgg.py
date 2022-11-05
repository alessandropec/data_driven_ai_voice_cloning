import torch.nn as nn
import torch

from torchvision import transforms

#assicurarsi che sia giusto per tutte le vgg, provare a modificare altri layer del classifier...
VGG_VERSIONS=	{"vgg11":{'fc6_input': 4096},
				 "vgg11_bn":{'fc6_input': 4096},
				 "vgg13":{'fc6_input': 4096},
				 "vgg13_bn":{'fc6_input': 4096},
				 "vgg16":{'fc6_input': 4096},
				 "vgg16_bn":{'fc6_input': 4096},
				 "vgg19":{'fc6_input': 4096},
				 "vgg19_bn":{'fc6_input': 4096},

				}

class SpVGG(nn.Module):
	def __init__(self,num_classes, pretrained=True,vgg_version="vgg11_bn",):
		super(SpVGG, self).__init__()
		self.pretrained=pretrained
		self.vgg_version=vgg_version
		self.fc6_input=VGG_VERSIONS[vgg_version]["fc6_input"]
		self.num_classes=num_classes

		self.model=torch.hub.load('pytorch/vision:v0.10.0', self.vgg_version, pretrained=self.pretrained)
		self.model.classifier[6]= nn.Linear(self.fc6_input, self.num_classes)
		
		
		
	   
	def forward(self, x):
		output = self.model(x)
		return output

def vgg_prepro(input):
	input_image,label=input
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	return preprocess(input_image),label