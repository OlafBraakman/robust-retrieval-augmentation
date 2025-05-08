# -*- coding: utf-8 -*-
# Based on https://github.com/Barchid/RGBD-Seg/blob/master/dataloaders/sunrgbd/sunrgbd.py

# Default SUNRGB dataset description
class SUNRBDBase:

    SPLITS = ['train', 'test']
    
    CAMERAS = ['realsense', 'kv2', 'kv1', 'xtion']

    SCENES = [
        'bathroom', 'bedroom', 'classroom', 'computer_room', 'conference_room', 
        'corridor', 'dining_area', 'dining_room', 'discussion_area', 
        'furniture_store', 'home_office', 'kitchen', 'lab', 'lecture_theatre', 
        'library', 'living_room', 'office', 'rest_space', 'study_space'
    ]

    SCENE_MAP = {
        'bathroom': 0, 'bedroom': 1, 'classroom': 2, 'computer_room': 3,
        'conference_room': 4, 'corridor': 5, 'dining_area': 6, 'dining_room': 7,
        'discussion_area': 8, 'furniture_store': 9, 'home_office': 10, 'kitchen': 11,
        'lab': 12, 'lecture_theatre': 13, 'library': 14, 'living_room': 15, 'office': 16,
        'rest_space': 17, 'study_space': 18
    }