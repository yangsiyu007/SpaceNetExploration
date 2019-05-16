import os
import random
import shutil


# This script splits all available images and labels at root_path into train, val and test sets in
# the ratio 0.7:0.15:0.15, and then copy each image label pair to the correct folders, specified
# by the outputs dictionary.
# The same file structure is maintained in this process so that the SpaceNet utilities code can be
# applied to each folder (train, val, test) without modification.
# This is necessary even though the SpaceNet utilities code also splits the data into trainval and test
# as it creates mask annotations from polygon labels, because that process only split the data after smaller chips are created from larger, raw images with some overlap.


root_path = '~/building_extraction/raw_data/'

image_dir_path = os.path.join(root_path, 'AOI_2_Vegas_Train', 'RGB-PanSharpen')
label_dir_path = os.path.join(root_path, 'AOI_2_Vegas_Train', 'geojson', 'buildings')

image_names = os.listdir(image_dir_path)
label_names = []

for image_name in image_names:
        if image_name.endswith('.tif'):
                parts = image_name.split('.')
                identifier = parts[0].split('RGB-PanSharpen_')[1]
                label_names.append('buildings_{}.geojson'.format(identifier))

# check if all corresponding geojson files exist
print('Starting checking all required geojson files exist')
for label_name in label_names:
        if not os.path.exists(os.path.join(label_dir_path, label_name)):
                print('{} does not exist'.format(label_name))

print('There are {} image files, {} geojson files'.format(len(image_names), len(label_names)))

# RGB-PanSharpen_AOI_2_Vegas_img4856.tif
# buildings_AOI_2_Vegas_img4867.geojson

images_labels = list(zip(image_names, label_names))
print('First pair before shuffle: {}'.format(images_labels[0]))
random.shuffle(images_labels) # in-place
print('First pair after shuffle: {}'.format(images_labels[0]))

train_len = int(len(images_labels) * 0.7)
val_len = int(len(images_labels) * 0.15)

splits = {}
splits['train'] = images_labels[:train_len]
splits['val'] = images_labels[train_len:train_len + val_len]
splits['test'] = images_labels[train_len + val_len:]

print('Resulting in {} train examples, {} val examples, {} test examples'.format(len(splits['train']), len(splits['val']), len(splits['test'])))

# create dirs
train_path = os.path.join(root_path, 'Vegas_processed_train')
val_path = os.path.join(root_path, 'Vegas_processed_val')
test_path = os.path.join(root_path, 'Vegas_processed_test')

outputs = {}
outputs['train_label'] = os.path.join(train_path, 'geojson', 'buildings')
outputs['train_image'] = os.path.join(train_path, 'RGB-PanSharpen')
outputs['val_label'] = os.path.join(val_path, 'geojson', 'buildings')
outputs['val_image'] = os.path.join(val_path, 'RGB-PanSharpen')
outputs['test_label'] = os.path.join(test_path, 'geojson', 'buildings')
outputs['test_image'] = os.path.join(test_path, 'RGB-PanSharpen')

for name, output_dir in outputs.items():
	os.makedirs(output_dir, exist_ok=True)

for split_name in ['train', 'val', 'test']:
	print('Copying to {} output dir'.format(split_name))
	for image_name, label_name in splits[split_name]:
		# copy to correct split file
		shutil.copy(os.path.join(image_dir_path, image_name), os.path.join(outputs['{}_image'.format(split_name)], image_name))
		shutil.copy(os.path.join(label_dir_path, label_name), os.path.join(outputs['{}_label'.format(split_name)], label_name))

print('End of split_train_val_test.py')
