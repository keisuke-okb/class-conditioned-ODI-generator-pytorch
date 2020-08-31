from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from PIL import Image

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, padding
from data import get_training_set, get_test_set

from tqdm import tqdm
import time

# Training settings
parser = argparse.ArgumentParser(description='class-conditioned-ODI-generator-pytorch')
parser.add_argument('--dataset', type=str, default="dataset", required=False, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--g_ch', type=int, default=128, help='generator channels in first conv layer')
parser.add_argument('--d_ch', type=int, default=128, help='discriminator channels in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--padding', action='store_true', default=False, help='use padding method?')
parser.add_argument('--save_interval', type=int, default=10,  help='interval epoch of network weight saving')
parser.add_argument('--graph_save_while_training', action='store_true', default=False, help='save loss graph while training?')
parser.add_argument('--output_path', type=str, default="sample", help='name of output directory')
opt = parser.parse_args()

root_path = "../"
class_name_array = ["arena","balcony","beach","boat_deck","bridge","cemetery",
"coast","desert","field","forest","highway","jetty","lawn","mountain","park",
"parking_lot","patio","plaza_courtyard","ruin","sports_field","street",
"swimming_pool","train_station_or_track","wharf"]

def main():

	print(f"epoch: {opt.niter+opt.niter_decay}")
	print(f"cuda: {opt.cuda}")
	print(f"dataset: {opt.dataset}")
	print(f"output: {opt.output_path}")

	if opt.cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")

	cudnn.benchmark = True

	torch.manual_seed(opt.seed)
	if opt.cuda:
		torch.cuda.manual_seed(opt.seed)

	print('Loading datasets')
	train_set = get_training_set(root_path + opt.dataset, opt.direction)
	test_set = get_test_set(root_path + opt.dataset, opt.direction)

	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

	device = torch.device("cuda:0" if opt.cuda else "cpu")

	print('Building models')
	net_g = define_G(opt.input_nc, opt.output_nc, opt.g_ch, len(class_name_array), 'batch', False, 'normal', 0.02, gpu_id=device)
	net_d = define_D(opt.input_nc + opt.output_nc, opt.d_ch, len(class_name_array), 'basic', gpu_id=device)

	criterionGAN = GANLoss().to(device)
	criterionL1 = nn.L1Loss().to(device)
	criterionMSE = nn.MSELoss().to(device)

	# setup optimizer
	optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	net_g_scheduler = get_scheduler(optimizer_g, opt)
	net_d_scheduler = get_scheduler(optimizer_d, opt)

	start_time = time.time()

	#save loss
	G_loss_array = []
	D_loss_array = []
	epoch_array = []

	for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), desc="Epoch"):
		# train
		loss_g_sum = 0
		loss_d_sum = 0
		for iteration, batch in enumerate(tqdm(training_data_loader, desc="Batch"), 1):
			# forward
			real_a, real_b, class_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3][0]
			fake_b = net_g(real_a, class_label)

			######################
			# (1) Update D network
			######################

			optimizer_d.zero_grad()
			
			# train with fake
			if opt.padding:
				real_a_for_d = padding(real_a)
				real_b_for_d = padding(real_b)
				fake_b_for_d = padding(fake_b)
			else:
				real_a_for_d = real_a
				real_b_for_d = real_b
				fake_b_for_d = fake_b
			
			fake_ab = torch.cat((real_a_for_d, fake_b_for_d), 1)
			pred_fake = net_d.forward(fake_ab.detach(), class_label)
			loss_d_fake = criterionGAN(pred_fake, False)

			# train with real
			real_ab = torch.cat((real_a_for_d, real_b_for_d), 1)
			pred_real = net_d.forward(real_ab, class_label)
			loss_d_real = criterionGAN(pred_real, True)
			
			# Combined D loss
			loss_d = (loss_d_fake + loss_d_real) * 0.5

			loss_d.backward()
		   
			optimizer_d.step()

			######################
			# (2) Update G network
			######################

			optimizer_g.zero_grad()

			# First, G(A) should fake the discriminator
			fake_ab = torch.cat((real_a_for_d, fake_b_for_d), 1)
			pred_fake = net_d.forward(fake_ab, class_label)
			loss_g_gan = criterionGAN(pred_fake, True)

			# Second, G(A) = B
			loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
			
			loss_g = loss_g_gan + loss_g_l1
			
			loss_g.backward()

			optimizer_g.step()
			loss_d_sum += loss_d.item()
			loss_g_sum += loss_g.item()

		update_learning_rate(net_g_scheduler, optimizer_g)
		update_learning_rate(net_d_scheduler, optimizer_d)
		
		# test
		avg_psnr = 0
		dst = Image.new('RGB', (512*4, 256*4))
		n = 0
		for batch in tqdm(testing_data_loader, desc="Batch"):
			input, target, class_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3][0]

			prediction = net_g(input, class_label)
			mse = criterionMSE(prediction, target)
			psnr = 10 * log10(1 / mse.item())
			avg_psnr += psnr
			
			n += 1
			if n <= 16:
				#make test preview
				out_img = prediction.detach().squeeze(0).cpu()
				image_numpy = out_img.float().numpy()
				image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
				image_numpy = image_numpy.clip(0, 255)
				image_numpy = image_numpy.astype(np.uint8)
				image_pil = Image.fromarray(image_numpy)
				dst.paste(image_pil, ((n-1)%4*512, (n-1)//4*256))
				
		if not os.path.exists("results"):
			os.mkdir("results")
		if not os.path.exists(os.path.join("results", opt.output_path)):
			os.mkdir(os.path.join("results", opt.output_path))
		dst.save(f"results/{opt.output_path}/epoch{epoch}_test_preview.jpg")
		
		epoch_array += [epoch]
		G_loss_array += [loss_g_sum/len(training_data_loader)]
		D_loss_array += [loss_d_sum/len(training_data_loader)]
		
		if opt.graph_save_while_training and len(epoch_array) > 1:
			output_graph(epoch_array, G_loss_array, D_loss_array, False)
		
		#checkpoint
		if epoch % opt.save_interval == 0:
			if not os.path.exists("checkpoint"):
				os.mkdir("checkpoint")
			if not os.path.exists(os.path.join("checkpoint", opt.output_path)):
				os.mkdir(os.path.join("checkpoint", opt.output_path))
			net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.output_path, epoch)
			net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.output_path, epoch)
			torch.save(net_g, net_g_model_out_path)
			torch.save(net_d, net_d_model_out_path)

	#save the latest net
	if not os.path.exists("checkpoint"):
		os.mkdir("checkpoint")
	if not os.path.exists(os.path.join("checkpoint", opt.output_path)):
		os.mkdir(os.path.join("checkpoint", opt.output_path))
	net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.output_path, opt.niter + opt.niter_decay)
	net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.output_path, opt.niter + opt.niter_decay)
	torch.save(net_g, net_g_model_out_path)
	torch.save(net_d, net_d_model_out_path)
	print("\nCheckpoint saved to {}".format("checkpoint/" + opt.output_path))

	# output loss graph
	output_graph(epoch_array, G_loss_array, D_loss_array)

	# finish training
	now_time = time.time()
	t = now_time - start_time
	print(f"Training time: {t/60:.1f}m")

def output_graph(epoch_array, G_loss_array, D_loss_array, show=True):
	# y = f(x)
	x = epoch_array
	y1 = G_loss_array
	y2 = D_loss_array

	# figure
	fig = plt.figure()
	fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,5))
	# plot
	axL.plot(x, y1, linestyle='-', color='r', label='Generator Loss')
	axR.plot(x, y2, linestyle='-', color='b', label='Discriminator Loss')

	# x axis
	axL.set_xlim([1, max(x)])
	axR.set_xlim([1, max(x)])

	axL.set_xlabel('Epoch')
	axR.set_xlabel('Epoch')

	# y axis
	# plt.ylim([0, 1])
	axL.set_ylabel('Loss')
	axR.set_ylabel('Loss')
	axL.set_ylim([0, max(y1)])
	axR.set_ylim([0, max(y2)])

	# legend and title
	axL.legend(loc='best')
	axR.legend(loc='best')

	axL.set_title('PyTorch Training Loss(G)')
	axR.set_title('PyTorch Training Loss(D)')

	# save as png
	plt.savefig("results/" + opt.output_path + '/loss_figure.png')
	if show:
		print("figure saved")

if __name__=='__main__':
	main()