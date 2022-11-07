from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from general_functions.generic_dataset import GenericDataset
import os

class DETRAC(GenericDataset):
    num_categories = 4
    default_resolution = [544, 960]
    class_name = ['car', 'bus', 'van', 'others']
    max_objs = 256
    cat_ids = {1: 1, 2: 2, 3: 3, 4: 4}

    def __init__(self, opt, split):
        # assert (opt.custom_dataset_img_path != '') and \
        #        (opt.custom_dataset_ann_path != '') and \
        #        (opt.num_classes != -1) and \
        #        (opt.input_h != -1) and (opt.input_w != -1), \
        #     'The following arguments must be specified for custom datasets: ' + \
        #     'custom_dataset_img_path, custom_dataset_ann_path, num_classes, ' + \
        #     'input_h, input_w.'
        # img_folders = {'train': 'Insight-MVT_Annotation_Train', 'test': 'Insight-MVT_Annotation_Test', 
        #                'grad': 'gradtrain', 'fine': 'finetune', 'fval': 'finetest'}
        # ann_folders = {'train': 'DETRAC-Train-Annotations-XML', 'test': 'DETRAC-Test-Annotations-XML'}

        # data_dir = opt.data_dir + 'DETRAC/'
        data_dir = opt.data_dir
        # img_dir = os.path.join(data_dir, img_folders[split])
        img_dir = data_dir + split + '/'
        # ann_dir = os.path.join(data_dir, '{}.json'.format(img_folders[split]))
        ann_dir = data_dir + 'annotations/' + split + '.json'
        # ann_dir = os.path.join(anno_dir, '{}.json'.format(img_folders[split]))
        self.num_categories = 4
        self.class_name = ['car', 'bus', 'van', 'others']
        self.default_resolution = [opt.input_h, opt.input_w]
        # self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}
        self.cat_ids = {1: 1, 2: 2, 3: 3, 4: 4}

        self.images = None
        # load image list and coco
        super(DETRAC, self).__init__(opt, split, ann_dir, img_dir)

        self.num_samples = len(self.images)
        # print('Loaded Custom dataset {} samples'.format(self.num_samples))
        print('Loaded DETRAC dataset {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    # def run_eval(self, results, save_dir):
    #     pass
