import requests
import json

def download_imagenet_classes():
    """ImageNet 1000개 클래스명을 다운로드"""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        classes = response.text.strip().split('\n')
        return classes
    else:
        print(f"다운로드 실패: {response.status_code}")
        return None

def get_imagenet_classes_local():
    """로컬에서 ImageNet 클래스명 반환 (일부 예시)"""
    classes = [
        "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
        "electric_ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting",
        "robin", "bulbul", "jay", "magpie", "chickadee",
        "water_ouzel", "kite", "bald_eagle", "vulture", "great_grey_owl",
        "European_fire_salamander", "common_newt", "eft", "spotted_salamander", "axolotl",
        "bullfrog", "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle",
        "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana",
        "American_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard",
        "Gila_monster", "green_lizard", "African_chameleon", "Komodo_dragon", "African_crocodile",
        "American_alligator", "triceratops", "thunder_snake", "ringneck_snake", "hognose_snake",
        "green_snake", "king_snake", "garter_snake", "water_snake", "vine_snake",
        "night_snake", "boa_constrictor", "rock_python", "Indian_cobra", "green_mamba",
        "sea_snake", "horned_viper", "diamondback", "sidewinder", "trilobite",
        "harvestman", "scorpion", "black_and_gold_garden_spider", "barn_spider", "garden_spider",
        "black_widow", "tarantula", "wolf_spider", "tick", "centipede",
        "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_chicken", "peacock",
        "quail", "partridge", "African_grey", "macaw", "sulphur-crested_cockatoo",
        "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird",
        "jacamar", "toucan", "drake", "red-breasted_merganser", "goose",
        "black_swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea_anemone", "brain_coral"
    ]
    return classes

def show_imagenet_categories():
    """ImageNet 주요 카테고리 분류"""
    categories = {
        "동물": {
            "포유류": ["dog", "cat", "horse", "cow", "sheep", "pig", "elephant", "bear", "zebra", "giraffe"],
            "조류": ["eagle", "owl", "parrot", "robin", "peacock", "flamingo", "pelican", "ostrich"],
            "어류": ["goldfish", "shark", "stingray", "tuna", "salmon"],
            "파충류": ["snake", "lizard", "turtle", "crocodile", "iguana"],
            "곤충": ["butterfly", "bee", "ant", "spider", "beetle"]
        },
        "식물": {
            "꽃": ["daisy", "rose", "sunflower", "tulip", "orchid"],
            "과일": ["apple", "banana", "orange", "strawberry", "pineapple"],
            "채소": ["broccoli", "carrot", "corn", "bell_pepper", "cabbage"]
        },
        "사물": {
            "차량": ["car", "truck", "motorcycle", "bicycle", "airplane", "ship"],
            "가전": ["refrigerator", "microwave", "washing_machine", "television", "computer"],
            "가구": ["chair", "table", "bed", "sofa", "wardrobe"],
            "도구": ["hammer", "screwdriver", "wrench", "scissors", "knife"]
        },
        "음식": {
            "요리": ["pizza", "hamburger", "hot_dog", "sandwich", "taco"],
            "음료": ["coffee", "tea", "wine", "beer", "cocktail"],
            "디저트": ["cake", "ice_cream", "cookie", "donut", "chocolate"]
        }
    }
    
    return categories

def search_imagenet_class(query, classes):
    """특정 키워드로 ImageNet 클래스 검색"""
    query = query.lower()
    matches = []
    
    for i, class_name in enumerate(classes):
        if query in class_name.lower():
            matches.append((i, class_name))
    
    return matches

if __name__ == "__main__":
    print("=== ImageNet 클래스 다운로드 시도 ===")
    classes = download_imagenet_classes()
    
    if classes:
        print(f"총 {len(classes)}개 클래스 다운로드 완료")
        print("\n처음 20개 클래스:")
        for i, class_name in enumerate(classes[:20]):
            print(f"{i:3d}: {class_name}")
        
        print("\n마지막 10개 클래스:")
        for i, class_name in enumerate(classes[-10:], start=len(classes)-10):
            print(f"{i:3d}: {class_name}")
            
        # 동물 관련 클래스 검색
        print("\n=== '개' 관련 클래스들 ===")
        dog_classes = search_imagenet_class("dog", classes)
        for idx, name in dog_classes:
            print(f"{idx:3d}: {name}")
            
        # 차량 관련 클래스 검색
        print("\n=== '차량' 관련 클래스들 ===")
        car_classes = search_imagenet_class("car", classes)
        for idx, name in car_classes:
            print(f"{idx:3d}: {name}")
            
    else:
        print("온라인 다운로드 실패, 로컬 예시 사용")
        local_classes = get_imagenet_classes_local()
        for i, class_name in enumerate(local_classes):
            print(f"{i:3d}: {class_name}")
    
    print("\n=== ImageNet 주요 카테고리 ===")
    categories = show_imagenet_categories()
    for main_cat, sub_cats in categories.items():
        print(f"\n{main_cat}:")
        for sub_cat, examples in sub_cats.items():
            print(f"  {sub_cat}: {', '.join(examples[:5])}") 