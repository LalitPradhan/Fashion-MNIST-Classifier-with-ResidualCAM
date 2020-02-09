import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
	'Cvgg16_bn',
]


model_urls = {
	'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class CVGG(nn.Module):

	def __init__(self, features, num_classes=10, init_weights=True):
		super(CVGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.doGAP = torch.nn.Dropout(0.3)
		self.fcGAP = nn.Linear(512, num_classes)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x1 = torch.flatten(x, 1)
		x1 = self.classifier(x1)
		x2 = F.avg_pool2d(x, kernel_size=7)
		x2 = self.doGAP(x2)
		x2 = x2.view(x2.size(0), -1)
		x2 = self.fcGAP(x2)
		x = x1*x2
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfgs = {
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


def C_vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
	if pretrained:
		kwargs['init_weights'] = False
	model = CVGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls[arch],
											  progress=progress)
		state_dict['fcGAP.weight'] = nn.init.normal_(torch.randn(10, 512), 0, 0.01)
		state_dict['fcGAP.bias'] = nn.init.constant_(torch.randn(10), 0)
		state_dict['classifier.6.weight'] = nn.init.normal_(torch.randn(10, 4096), 0, 0.01)
		state_dict['classifier.6.bias'] = nn.init.constant_(torch.randn(10), 0)
		model.load_state_dict(state_dict)
	return model


def Cvgg16_bn(pretrained=False, progress=True, **kwargs):
	r"""VGG 16-layer model (configuration "D") with batch normalization
	`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return C_vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)

