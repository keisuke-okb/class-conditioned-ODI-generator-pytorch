import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
	scheduler.step()
	lr = optimizer.param_groups[0]['lr']
	#print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		#U-Net
		net.weight_init(mean=0.0, std=gain)

	#print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net.to(gpu_id)
	init_weights(net, init_type, gain=init_gain)
	return net


def define_G(input_nc, output_nc, ngf, n_class, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)
	net = generator_class_conditioned(ngf, n_class)
	return init_net(net, init_type, init_gain, gpu_id)
	
def define_D(input_nc, ndf, n_class, netD,
			 n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)
	net = discriminator_class_conditioned(ndf, n_class)
	return init_net(net, init_type, init_gain, gpu_id)

# =============== Define U-Net Generator / Discriminator <Class Conditioned> ===============
class generator_class_conditioned(nn.Module):
	# initializers
	def __init__(self, d=64, n_class=24):
		super(generator_class_conditioned, self).__init__()
		# Unet encoder
		# default: filter 4, stride 2, padding 1
		self.conv1 = nn.Conv2d(3, d, 3, 1, 1)
		self.fc1 = nn.Linear(n_class, d)
		
		self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
		self.conv2_bn = nn.BatchNorm2d(d * 2)
		self.fc2 = nn.Linear(n_class, d * 2)
		
		self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
		self.conv3_bn = nn.BatchNorm2d(d * 4)
		self.fc3 = nn.Linear(n_class, d * 4)
		
		self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
		self.conv4_bn = nn.BatchNorm2d(d * 8)
		self.fc4 = nn.Linear(n_class, d * 8)
		
		self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
		self.conv5_bn = nn.BatchNorm2d(d * 8)
		self.fc5 = nn.Linear(n_class, d * 8)
		
		self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
		self.conv6_bn = nn.BatchNorm2d(d * 8)
		self.fc6 = nn.Linear(n_class, d * 8)
		
		self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
		self.conv7_bn = nn.BatchNorm2d(d * 8)
		self.fc7 = nn.Linear(n_class, d * 8)
		
		self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
		self.fc8 = nn.Linear(n_class, d * 8)

		self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
		self.deconv1_bn = nn.BatchNorm2d(d * 8)
		self.fcd1 = nn.Linear(n_class, d * 8)
		
		self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
		self.deconv2_bn = nn.BatchNorm2d(d * 8)
		self.fcd2 = nn.Linear(n_class, d * 8)
		
		self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
		self.deconv3_bn = nn.BatchNorm2d(d * 8)
		self.fcd3 = nn.Linear(n_class, d * 8)
		
		self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
		self.deconv4_bn = nn.BatchNorm2d(d * 8)
		self.fcd4 = nn.Linear(n_class, d * 8)
		
		self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
		self.deconv5_bn = nn.BatchNorm2d(d * 4)
		self.fcd5 = nn.Linear(n_class, d * 4)
		
		self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
		self.deconv6_bn = nn.BatchNorm2d(d * 2)
		self.fcd6 = nn.Linear(n_class, d * 2)
		
		self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
		self.deconv7_bn = nn.BatchNorm2d(d)
		self.fcd7 = nn.Linear(n_class, d)
		
		self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 3, 1, 1)
		self.fcd8 = nn.Linear(n_class, 3)

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, input, class_label):

		e1 = (self.conv1(input).permute(0,2,3,1) * self.fc1(class_label)).permute(0,3,1,2)
		e2 = self.conv2_bn((self.conv2(F.leaky_relu(e1, 0.2)).permute(0,2,3,1) * self.fc2(class_label)).permute(0,3,1,2))
		e3 = self.conv3_bn((self.conv3(F.leaky_relu(e2, 0.2)).permute(0,2,3,1) * self.fc3(class_label)).permute(0,3,1,2))
		e4 = self.conv4_bn((self.conv4(F.leaky_relu(e3, 0.2)).permute(0,2,3,1) * self.fc4(class_label)).permute(0,3,1,2))
		e5 = self.conv5_bn((self.conv5(F.leaky_relu(e4, 0.2)).permute(0,2,3,1) * self.fc5(class_label)).permute(0,3,1,2))
		e6 = self.conv6_bn((self.conv6(F.leaky_relu(e5, 0.2)).permute(0,2,3,1) * self.fc6(class_label)).permute(0,3,1,2))
		e7 = self.conv7_bn((self.conv7(F.leaky_relu(e6, 0.2)).permute(0,2,3,1) * self.fc7(class_label)).permute(0,3,1,2))
		e8 = (self.conv8(F.leaky_relu(e7, 0.2)).permute(0,2,3,1) * self.fc8(class_label)).permute(0,3,1,2)
		
		d1 = F.dropout(self.deconv1_bn((self.deconv1(F.relu(e8)).permute(0,2,3,1) * self.fcd1(class_label)).permute(0,3,1,2)), 0.5, training=True)
		d1 = torch.cat([d1, e7], 1)
		d2 = F.dropout(self.deconv2_bn((self.deconv2(F.relu(d1)).permute(0,2,3,1) * self.fcd2(class_label)).permute(0,3,1,2)), 0.5, training=True)
		d2 = torch.cat([d2, e6], 1)
		d3 = F.dropout(self.deconv3_bn((self.deconv3(F.relu(d2)).permute(0,2,3,1) * self.fcd3(class_label)).permute(0,3,1,2)), 0.5, training=True)
		d3 = torch.cat([d3, e5], 1)
		d4 = self.deconv4_bn((self.deconv4(F.relu(d3)).permute(0,2,3,1) * self.fcd4(class_label)).permute(0,3,1,2))
		d4 = torch.cat([d4, e4], 1)
		d5 = self.deconv5_bn((self.deconv5(F.relu(d4)).permute(0,2,3,1) * self.fcd5(class_label)).permute(0,3,1,2))
		d5 = torch.cat([d5, e3], 1)
		d6 = self.deconv6_bn((self.deconv6(F.relu(d5)).permute(0,2,3,1) * self.fcd6(class_label)).permute(0,3,1,2))
		d6 = torch.cat([d6, e2], 1)
		d7 = self.deconv7_bn((self.deconv7(F.relu(d6)).permute(0,2,3,1) * self.fcd7(class_label)).permute(0,3,1,2))
		d7 = torch.cat([d7, e1], 1)
		d8 = (self.deconv8(F.relu(d7)).permute(0,2,3,1) * self.fcd8(class_label)).permute(0,3,1,2)
		# o = torch.tanh(d8)

		return d8

class discriminator_class_conditioned(nn.Module):
	# initializers
	def __init__(self, d=64, n_class=24):
		super(discriminator_class_conditioned, self).__init__()
		self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
		self.fc1 = nn.Linear(n_class, d)
		
		self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
		self.conv2_bn = nn.BatchNorm2d(d * 2)
		self.fc2 = nn.Linear(n_class, d * 2)
		
		self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
		self.conv3_bn = nn.BatchNorm2d(d * 4)
		self.fc3 = nn.Linear(n_class, d * 4)
		
		self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
		self.conv4_bn = nn.BatchNorm2d(d * 8)
		self.fc4 = nn.Linear(n_class, d * 8)
		
		self.conv5 = nn.Conv2d(d * 8, 1, 3, 1, 1)
		self.fc5 = nn.Linear(n_class, 1)

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, input, class_label):
		x = input
		x = F.leaky_relu((self.conv1(x).permute(0,2,3,1) * self.fc1(class_label)).permute(0,3,1,2), 0.2)
		x = F.leaky_relu(self.conv2_bn((self.conv2(x).permute(0,2,3,1) * self.fc2(class_label)).permute(0,3,1,2)), 0.2)
		x = F.leaky_relu(self.conv3_bn((self.conv3(x).permute(0,2,3,1) * self.fc3(class_label)).permute(0,3,1,2)), 0.2)
		x = F.leaky_relu(self.conv4_bn((self.conv4(x).permute(0,2,3,1) * self.fc4(class_label)).permute(0,3,1,2)), 0.2)
		x = (self.conv5(x).permute(0,2,3,1) * self.fc5(class_label)).permute(0,3,1,2)

		return x

def normal_init(m, mean, std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()

class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

# Padding method
def padding(input):
	x_0 = torch.einsum("ijkl->klji", input)
	left = x_0[:,0:50,:]
	right = x_0[:,461:511,:]
	upper_tmp = x_0[0,0,:]
	bottom_tmp = x_0[255,0,:]
	upper = upper_tmp.repeat(1,612,1,1)
	bottom = bottom_tmp.repeat(1,612,1,1)
	x_0 = torch.cat((x_0, left), 1)
	x_0 = torch.cat((right, x_0), 1)
	x_0 = torch.cat((upper, x_0), 0)
	x_0 = torch.cat((x_0, bottom), 0)
	x_0 = torch.einsum("ijkl->lkij", x_0)
	return x_0