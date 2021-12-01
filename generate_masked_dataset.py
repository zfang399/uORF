# /home/mihir/Documents/phd_projects/slots/uorf_data_gen/image_generation/rendered_images/new_clevr/*
import glob
import shutil
import os
import numpy as np
import shutil
import ipdb
import sys
import cv2
st = ipdb.set_trace
val_arg = sys.argv[1]
files = glob.glob(val_arg+ "/*_mask.png")
save_folder = val_arg + "_masked"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir(save_folder)

for idx, mask_file in enumerate(files):
	print(mask_file,idx)
	try:
		rgb_file = mask_file.replace("_mask.png",".png")
		rt_file = mask_file.replace("_mask.png","_RT.txt")
		mask_image = cv2.imread(mask_file)
		rgb_image = cv2.imread(rgb_file)
		# mask_image == [64,64,64]
		bin_mask = mask_image != [64,64,64]
		bin_mask = bin_mask.astype(np.uint8)
		rgb_image_masked = bin_mask*rgb_image
		rgb_masked_file = mask_file.replace("_mask.png",".png")
		rgb_masked_file = rgb_masked_file.replace(val_arg,save_folder)
		cv2.imwrite(rgb_masked_file, rgb_image_masked)
		shutil.copy(rt_file, rt_file.replace(val_arg,save_folder))
	except Exception:
		print("Error",mask_file)
		pass

	# st()
# 	# st()
# 	# rgb_file = mask_file.replace("_mask.png","_RT.txt")
# 	# st()
# print("hello")