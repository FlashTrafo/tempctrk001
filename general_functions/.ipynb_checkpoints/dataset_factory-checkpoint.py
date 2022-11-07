from general_functions.datasets.detracloader import DETRAC

# dataset_factory = {
#   'custom': CustomDataset,
#   'coco': COCO,
#   'kitti': KITTI,
#   'coco_hp': COCOHP,
#   'mot': MOT,
#   'nuscenes': nuScenes,
#   'crowdhuman': CrowdHuman,
#   'kitti_tracking': KITTITracking,
# }

dataset_factory = {
    'detrac': DETRAC
    # 'custom': CustomDataset
}

def get_dataset(dataset):
    return dataset_factory[dataset]