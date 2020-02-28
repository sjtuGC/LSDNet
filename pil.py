from PIL import Image
import glob
#import shutil
import numpy as np

"""
num = 0
for name in glob.glob('/home/chengguan/lip_data/DlbProcess/annotation/*.png'):
	im = Image.open(name)
	#width, height = im.size
	im = im.resize((224,112),Image.BILINEAR)
	print(name," ",im.size)
	new_name = name + ".224*112"
	im.save(new_name,'png')
	num += 1
"""

im = Image.open('/home/chengguan/lip_data/DlbProcess/annotation/user9_random_9863_031.jpg.224*112')
print(im.size)
 
image = np.array(im,dtype=np.uint8)
for i in range(112):
	print(image[i])
print(image)
print(len(image[0]))
#print(num)

