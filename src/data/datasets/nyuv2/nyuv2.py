class NYUv2Base:
    SPLITS = ['train', 'test']
    SPLIT_FILELIST_FILENAMES = {SPLITS[0]: 'train.txt', SPLITS[1]: 'test.txt'}
    SPLIT_DIRS = {SPLITS[0]: 'train', SPLITS[1]: 'test'}

    DEPTH_DIR = 'depth'
    DEPTH_RAW_DIR = 'depth_raw'
    RGB_DIR = 'rgb'
    SCENE_DIR = 'scene'

    LABELS_DIR_FMT = 'labels_{:d}'
    LABELS_COLORED_DIR_FMT = 'labels_{:d}_colored'

    CAMERAS = ['kv1']

    # Classification
    SCENES = [
        'bedroom', 'kitchen', 'living_room', 'bathroom', 'dining_room', 'office', 'home_office', 'classroom', 'bookstore', 
        'indoor_balcony', 'excercise_room', 'printer_room', 'laundry_room',
        'dinette', 'foyer', 'conference_room', 'cafe', 'home_storage',
        'student_lounge', 'computer_lab', 'basement', 'study_room',
        'office_kitchen', 'reception_room', 'study', 'furniture_store',
        'playroom'  
    ]

    REDUCED_SCENES = [
        'bedroom', 'kitchen', 'living_room', 'bathroom', 'dining_room', 'office', 'home_office', 'classroom', 'bookstore', 'others'
    ]

    SCENE_MAP = {
        'bedroom': 0, 'kitchen': 1, 'living_room': 2, 'bathroom': 3, 'dining_room': 4, 'office': 5, 'home_office': 6, 'classroom': 7, 'bookstore': 8, 'others': 9
    }

    @classmethod
    def map_scene(cls, x):
        if x in cls.REDUCED_SCENES:
            return x
        return "others"
