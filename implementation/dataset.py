from os import listdir
from os.path import join

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file
import re

class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, direction):
		super(DatasetFromFolder, self).__init__()
		self.direction = direction
		self.a_path = join(image_dir, "base")
		self.b_path = join(image_dir, "label")
		self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

		transform_list = [transforms.ToTensor(),
						  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, index):
		a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
		b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
		#a = a.resize((286, 286), Image.BICUBIC)
		#b = b.resize((286, 286), Image.BICUBIC)
		a = transforms.ToTensor()(a)
		b = transforms.ToTensor()(b)
	
		a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
		b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
		
		class_name_array = ["arena","balcony","beach","boat_deck","bridge","cemetery",
		"coast","desert","field","forest","highway","jetty","lawn","mountain","park",
		"parking_lot","patio","plaza_courtyard","ruin","sports_field","street",
		"swimming_pool","train_station_or_track","wharf"]
		
		class_label = torch.zeros(len(class_name_array))
		class_name = re.sub(r'_\d{,5}.jpg', '', self.image_filenames[index])
		class_label[class_name_array.index(class_name)] = 1
		
		if self.direction == "a2b":
			return a, b, class_label, self.image_filenames[index]
		else:
			return b, a, class_label, self.image_filenames[index]

	def __len__(self):
		return len(self.image_filenames)
