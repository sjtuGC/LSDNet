import math
import numpy as np
import torch
import torch.nn as nn
import itertools
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
			 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class TeacherNet(nn.Module):
	def __init__(self, block, layers, n_class=2, zero_init_residual=False):
		super(TeacherNet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
			bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		
		self.deconv4 = nn.ConvTranspose2d(in_channels=2048,out_channels=1024,kernel_size=(1,2),stride=(2,2),bias=False)
		self.deconv3 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv2 = nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=2,kernel_size=(4,4),stride=(4,4),bias=False)
		
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				#m.weight.data.zero_()
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels*m.in_channels
				m.weight.data.normal_(0,math.sqrt(2. / n))
				if m.bias is not None:
                                        m.bias.data.zero_()
			if isinstance(m,nn.Linear):
				m.weight.data.normal_(0, 1)

		for param in self.parameters():
			param.requires_grad = True



	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
			conv1x1(self.inplanes, planes * block.expansion, stride),
			nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x0 = x
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)
		h4 = self.deconv4(x4)+x3
		h3 = self.deconv3(h4)+x2
		h2 = self.deconv2(h3)+x0
		h1  = self.deconv1(h2)
		return h1



class StudentNet(nn.Module):
	def __init__(self, block, layers, n_class=2, zero_init_residual=False):
		super(StudentNet, self).__init__()
		self.inplanes = 32
		self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
			bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 32, layers[0])
		self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
		self.deconv4 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(1,2),stride=(2,2),bias=False)
		self.deconv3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(2,2),stride=(2,2),bias=False)
		self.deconv1 = nn.ConvTranspose2d(in_channels=32,out_channels=2,kernel_size=(4,4),stride=(4,4),bias=False)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				#m.weight.data.zero_()
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels*m.in_channels
				m.weight.data.normal_(0,math.sqrt(2. / n))
				if m.bias is not None:
                                        m.bias.data.zero_()
			if isinstance(m,nn.Linear):
				m.weight.data.normal_(0, 1)

		for param in self.parameters():
			param.requires_grad = True



	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
			conv1x1(self.inplanes, planes * block.expansion, stride),
			nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x0 = x
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)
		h4 = self.deconv4(x4)+x3
		h3 = self.deconv3(h4)+x2
		h2 = self.deconv2(h3)+x0
		h1  = self.deconv1(h2)
		return h1



