class EuroSATBase:

    SPLITS = ['train', 'test']
    
    CLASSES = [
        "Industrial",
        "Residential",
        "AnnualCrop",
        "PermanentCrop",
        "River",
        "SeaLake",
        "HerbaceousVegetation",
        "Highway",
        "Pasture",
        "Forest",
    ]

    CLASS_MAP = {
        "Industrial": 0,
        "Residential": 1,
        "AnnualCrop": 2,
        "PermanentCrop": 3,
        "River": 4,
        "SeaLake": 5,
        "HerbaceousVegetation": 6,
        "Highway": 7,
        "Pasture": 8,
        "Forest": 9,
    }