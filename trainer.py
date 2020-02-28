import datetime
import math
import os
import sys
import os.path as osp

import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import torch.nn as nn

def _fast_hist(label_true, label_pred, n_class):
	mask = (label_true >= 0) & (label_true < n_class)
	hist = np.bincount(
		n_class * label_true[mask].astype(int) +
		label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
	return hist

def cross_entropy2d(input, target, weight=None, size_average=False):
	# input: (n, c, h, w), target: (n, h, w)
	n, c, h, w = input.size()
	# log_p: (n, c, h, w)
	log_p = F.log_softmax(input, dim=1)
	# log_p: (n*h*w, c)
	log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
	log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
	log_p = log_p.view(-1, c)
	#print(log_p)
	# target: (n*h*w,)
	mask = target >= 0 
	target = target[mask]
	loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
	if loss is None:
		print("ERROR")
	if size_average:
		loss = loss / mask.data.sum()
	return loss

def distillation_loss(score_student,score_teacher,T=2.0):
	#print(score_student.size(),score_teacher.size())
	assert score_student.size()==score_teacher.size()
	score_teacher = score_teacher.detach()
	KD_loss = nn.KLDivLoss()(F.log_softmax(score_student/T, dim=1), 
				F.softmax(score_teacher/T, dim=1)) * ( T * T) 
	return KD_loss

def remedy_loss(score_student,score_teacher, target, Z=0.6):
	score_student = F.softmax(score_student,dim=1)
	score_teacher = F.softmax(score_teacher,dim=1)

	co_lip = (1-Z)*torch.ones_like(target).float()-score_teacher[:,1,:,:]
	margin_lip = torch.max(torch.zeros_like(target).float(),co_lip)
	proxy_lip = -torch.log(score_student[:,1,:,:])*margin_lip

	co_bg = score_teacher[:,1,:,:]-Z*torch.ones_like(target).float()
	margin_bg = torch.max(torch.zeros_like(target).float(),co_bg)
	proxy_bg = -torch.log(score_student[:,0,:,:])*margin_bg

	loss = torch.where(target>0, proxy_lip, proxy_bg)
	loss = torch.sum(loss)
	#print(loss)
	
	return loss

class Trainer(object):
	def __init__(self, cuda, model, optimizer,train_loader, val_loader, out,out_model, max_iter,size_average=False, interval_validate=None,log=None):
		self.cuda = cuda
		self.model1 = model[0]
		self.model2 = model[1]
		self.optim1 = optimizer[0]
		self.optim2 = optimizer[1]

		self.train_loader = train_loader
		self.val_loader = val_loader
		self.size_average = size_average

		if interval_validate is None:
			self.interval_validate = len(self.train_loader)
		else:
			self.interval_validate = interval_validate

		self.out = out
		self.out_model = out_model

		self.epoch = 0
		self.iteration = 0
		self.max_iter = max_iter
		self.best_mean_iu = 0
		self.log = log
		
	def label_accuracy_score(self,label_trues, label_preds, n_class):

		hist = np.zeros((n_class, n_class))
		for lt, lp in zip(label_trues, label_preds):
			hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
		acc = np.diag(hist).sum() / hist.sum()
		acc_cls = np.diag(hist) / hist.sum(axis=1)
		acc_cls = np.nanmean(acc_cls)
		iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
		mean_iu = np.nanmean(iu)
		freq = hist.sum(axis=1) / hist.sum()
		fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
		return acc, acc_cls, mean_iu, fwavacc
	
	def validate(self):
		print("Now Validating...")
		training = True
		n_class = len(self.val_loader.dataset.class_names)
		print(n_class)
		print(len(self.val_loader))	
		val_loss = 0
		visualizations = []
		label_trues, label_preds = [], []
		for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader),
								desc='Valid iteration=%d' % self.iteration, ncols=80,leave=False):
		
			if self.cuda:
				data, target = data.cuda(), target.cuda()
				data, target = Variable(data), Variable(target)
			score = self.model2(data)

			loss = cross_entropy2d(score, target,size_average=self.size_average)
			loss_data = float(loss.data[0])
			if np.isnan(loss_data):
				raise ValueError('loss is nan while validating')
				val_loss += loss_data / len(data)

			imgs = data.data.cpu()
			lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
			lbl_true = target.data.cpu()
			for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
				img, lt = self.val_loader.dataset.untransform(img, lt)
				label_trues.append(lt)
				label_preds.append(lp)
		metrics = self.label_accuracy_score(label_trues, label_preds, n_class)

		val_loss /= len(self.val_loader)
		

		mean_iu = metrics[2]
		is_best = mean_iu > self.best_mean_iu

		with open(self.out,"a+") as f:
			f.write("**********************")
			log_string = "Epoch"+str(self.epoch)+"Iteration"+str(self.iteration)+"\nIU = "+str(mean_iu)+"\n"
			f.write(log_string)
			metric_string = str(metrics[0])+" "+str(metrics[1])+" "+str(metrics[2])+" "+str(metrics[3])+"\n"
			f.write(metric_string)
			if is_best:
				f.write("INFO:Mean IU is the current best\n")
			f.close()

		if is_best:
			self.best_mean_iu = mean_iu
			torch.save(self.model2.state_dict(),self.out_model)			

		if training:
			pass
	
	def train(self):
		max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
		for epoch in tqdm.trange(self.epoch, max_epoch,desc='Training', ncols=80):
			self.epoch = epoch
			self.train_epoch()
			if self.iteration >= self.max_iter:
				break

	def train_epoch(self):
		#self.model.train()

		n_class = len(self.train_loader.dataset.class_names)

		for batch_idx, (data, target) in tqdm.tqdm(
							enumerate(self.train_loader), total=len(self.train_loader),
							desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
			iteration = batch_idx + self.epoch * len(self.train_loader)
			if self.iteration != 0 and (iteration - 1) != self.iteration:
				continue  # for resuming
			self.iteration = iteration

			if self.iteration % self.interval_validate == 0 and self.iteration>0:
				self.validate()
				pass
			#print(data.size())
			#print(target.size())

			if self.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)

			self.optim1.zero_grad()
			score_teacher = self.model1(data)
			loss_teacher = cross_entropy2d(score_teacher, target,size_average=self.size_average)
			loss_teacher /= len(data)
			#print(loss_teacher,type(loss_teacher))
			#print(loss_teacher.data,type(loss_teacher.data))
	
			loss_t = loss_teacher
			if np.isnan(loss_t.data.item()):
				raise ValueError('loss t is nan while training')

			self.optim2.zero_grad()
			score_student = self.model2(data)
			loss_student_1 = cross_entropy2d(score_student, target, size_average=self.size_average)
			loss_student_1 /= len(data)
			loss_s = loss_student_1
			loss_soft = distillation_loss(score_student,score_teacher,T=3.0)
			soft_weight = 10000
			loss_s += soft_weight*loss_soft

			loss_remedy = remedy_loss(score_student,score_teacher,target,Z=0.9)
			remedy_weight = 1
			loss_s += remedy_weight*loss_remedy
 
			if np.isnan(loss_s.data.item()):
				raise ValueError("loss s_1 is nana while training")

			print(loss_t.data.item(),loss_s.data.item(),soft_weight*loss_soft.data.item(),loss_remedy.data.item())
			loss_s.backward(retain_graph=True)
			self.optim2.step()
			
			loss_t.backward()
			self.optim1.step()
			# NEED TO BE WRITTEN
			score = score_teacher
			metrics = []
			lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
			lbl_true = target.data.cpu().numpy()
			metrics = np.mean(metrics, axis=0)

			if self.iteration >= self.max_iter:
				break
