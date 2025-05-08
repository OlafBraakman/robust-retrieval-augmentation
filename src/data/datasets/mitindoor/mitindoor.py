class MITIndoorBase:

    SPLITS = ["train", "test"]

    SCENES = [
        'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar',
        'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino',
        'children_room', 'church_inside', 'classroom', 'cloister',
        'closet', 'clothingstore', 'computerroom', 'concert_hall',
        'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator',
        'fastfood_restaurant', 'florist', 'gameroom', 'garage',
        'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom',
        'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden',
        'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom',
        'lobby', 'locker_room', 'mall', 'meeting_room', 'movietheater',
        'museum', 'nursery', 'office', 'operating_room', 'pantry',
        'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen',
        'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore',
        'trainstation', 'tv_studio', 'videostore', 'waitingroom',
        'warehouse', 'winecellar'
    ]

    SCENE_MAP = {
        'airport_inside': 0, 'artstudio': 1, 'auditorium': 2, 'bakery': 3, 'bar': 4,
        'bathroom': 5, 'bedroom': 6, 'bookstore': 7, 'bowling': 8, 'buffet': 9, 'casino': 10,
        'children_room': 11, 'church_inside': 12, 'classroom': 13, 'cloister': 14,
        'closet': 15, 'clothingstore': 16, 'computerroom': 17, 'concert_hall': 18,
        'corridor': 19, 'deli': 20, 'dentaloffice': 21, 'dining_room': 22, 'elevator': 23,
        'fastfood_restaurant': 24, 'florist': 25, 'gameroom': 26, 'garage': 27,
        'greenhouse': 28, 'grocerystore': 29, 'gym': 30, 'hairsalon': 31, 'hospitalroom': 32,
        'inside_bus': 33, 'inside_subway': 34, 'jewelleryshop': 35, 'kindergarden': 36,
        'kitchen': 37, 'laboratorywet': 38, 'laundromat': 39, 'library': 40, 'livingroom': 41,
        'lobby': 42, 'locker_room': 43, 'mall': 44, 'meeting_room': 45, 'movietheater': 46,
        'museum': 47, 'nursery': 48, 'office': 49, 'operating_room': 50, 'pantry': 51,
        'poolinside': 52, 'prisoncell': 53, 'restaurant': 54, 'restaurant_kitchen': 55,
        'shoeshop': 56, 'stairscase': 57, 'studiomusic': 58, 'subway': 59, 'toystore': 60,
        'trainstation': 61, 'tv_studio': 62, 'videostore': 63, 'waitingroom': 64,
        'warehouse': 65, 'winecellar': 66
    }