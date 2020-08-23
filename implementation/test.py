from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=False, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
opt = parser.parse_args()

#=====================SETTINGS======================
folder_name = "beach_padding_modified"
opt.dataset = "beach"
opt.nepochs = 300
#===================================================
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(folder_name, opt.nepochs)

net_g = torch.load(model_path).to(device)

test_dataset_root = "../datasets_t3t1/per_class_training/"

if opt.direction == "a2b":
	image_dir = f"{test_dataset_root}{opt.dataset}/test/base/"
else:
	image_dir = f"{test_dataset_root}{opt.dataset}/test/label/"

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
				  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
	img = load_img(image_dir + image_name)
	img = transform(img)
	input = img.unsqueeze(0).to(device)
	
	class_name_array = ["arena","balcony","beach","boat_deck","bridge","cemetery",
	"coast","desert","field","forest","highway","jetty","lawn","mountain","park",
	"parking_lot","patio","plaza_courtyard","ruin","sports_field","street",
	"swimming_pool","train_station_or_track","wharf"]
	class_label = torch.zeros(24)
	class_name = re.sub(r'_\d{,5}.jpg', '', image_name)
	class_label[class_name_array.index(class_name)] = 1
	
	out = net_g(input, class_label)
	out_img = out.detach().squeeze(0).cpu()

	if not os.path.exists(os.path.join("generate", folder_name)):
		os.makedirs(os.path.join("generate", folder_name))
	save_img(out_img, "generate/{}/{}".format(folder_name, image_name))
