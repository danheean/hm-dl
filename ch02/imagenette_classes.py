# Imagenette 클래스들을 ImageNet 클래스명으로 매핑
import os

def get_imagenette_class_mapping():
    """Imagenette 클래스 ID를 실제 ImageNet 클래스명으로 매핑"""
    
    # 현재 train 디렉토리에 있는 클래스들
    imagenette_classes = {
        'n01440764': 'tench',  # 물고기
        'n02102040': 'English_springer',  # 개 (English Springer Spaniel)
        'n02979186': 'cassette_player',  # 카세트 플레이어
        'n03000684': 'chain_saw',  # 체인톱
        'n03028079': 'church',  # 교회
        'n03394916': 'French_horn',  # 프렌치 혼
        'n03417042': 'garbage_truck',  # 쓰레기트럭
        'n03425413': 'gas_pump',  # 주유소
        'n03445777': 'golf_ball',  # 골프공
        'n03888257': 'parachute'  # 낙하산
    }
    
    return imagenette_classes

def get_imagenet_class_details():
    """ImageNet 클래스들의 상세 정보"""
    
    class_details = {
        'tench': {
            'korean': '텐치 (물고기)',
            'category': '어류',
            'description': '잉어과 물고기, 유럽 민물고기',
            'difficulty': '중간',
            'alexnet_good': True
        },
        'English_springer': {
            'korean': '잉글리시 스프링거 스패니얼',
            'category': '개',
            'description': '중형 사냥개, 털이 길고 귀가 늘어진 개',
            'difficulty': '쉬움',
            'alexnet_good': True
        },
        'cassette_player': {
            'korean': '카세트 플레이어',
            'category': '전자기기',
            'description': '구식 음향기기, 카세트 테이프 재생기',
            'difficulty': '어려움',
            'alexnet_good': False
        },
        'chain_saw': {
            'korean': '체인톱',
            'category': '도구',
            'description': '전동 벌목용 톱, 체인이 달린 톱',
            'difficulty': '중간',
            'alexnet_good': True
        },
        'church': {
            'korean': '교회',
            'category': '건축물',
            'description': '종교 건축물, 첨탑이 있는 건물',
            'difficulty': '중간',
            'alexnet_good': True
        },
        'French_horn': {
            'korean': '프렌치 혼',
            'category': '악기',
            'description': '관악기, 나선형 금관악기',
            'difficulty': '어려움',
            'alexnet_good': False
        },
        'garbage_truck': {
            'korean': '쓰레기트럭',
            'category': '차량',
            'description': '쓰레기 수거용 트럭',
            'difficulty': '중간',
            'alexnet_good': True
        },
        'gas_pump': {
            'korean': '주유기',
            'category': '시설물',
            'description': '주유소의 연료 공급 장치',
            'difficulty': '중간',
            'alexnet_good': True
        },
        'golf_ball': {
            'korean': '골프공',
            'category': '스포츠용품',
            'description': '골프용 흰색 공, 딤플이 있는 공',
            'difficulty': '어려움',
            'alexnet_good': False
        },
        'parachute': {
            'korean': '낙하산',
            'category': '장비',
            'description': '낙하용 천, 스카이다이빙 장비',
            'difficulty': '중간',
            'alexnet_good': True
        }
    }
    
    return class_details

def recommend_classes_for_alexnet():
    """AlexNet 학습에 추천하는 클래스들"""
    
    class_details = get_imagenet_class_details()
    
    # AlexNet에 좋은 클래스들
    good_classes = []
    difficult_classes = []
    
    for class_name, details in class_details.items():
        if details['alexnet_good']:
            good_classes.append({
                'class': class_name,
                'korean': details['korean'],
                'category': details['category'],
                'reason': '형태가 뚜렷하고 학습하기 좋음'
            })
        else:
            difficult_classes.append({
                'class': class_name,
                'korean': details['korean'],
                'category': details['category'],
                'reason': '형태가 복잡하거나 작은 물체'
            })
    
    return good_classes, difficult_classes

def analyze_current_dataset():
    """현재 데이터셋 분석"""
    
    imagenette_mapping = get_imagenette_class_mapping()
    class_details = get_imagenet_class_details()
    
    print("=== 현재 데이터셋 분석 ===")
    print(f"총 클래스 수: {len(imagenette_mapping)}개")
    print("\n클래스별 정보:")
    
    for class_id, class_name in imagenette_mapping.items():
        details = class_details[class_name]
        print(f"\n{class_id} → {class_name}")
        print(f"  한국어: {details['korean']}")
        print(f"  카테고리: {details['category']}")
        print(f"  설명: {details['description']}")
        print(f"  AlexNet 학습 난이도: {details['difficulty']}")
        print(f"  추천 여부: {'✓' if details['alexnet_good'] else '✗'}")

if __name__ == "__main__":
    # 현재 데이터셋 분석
    analyze_current_dataset()
    
    print("\n" + "="*50)
    
    # AlexNet 학습 추천 클래스
    good_classes, difficult_classes = recommend_classes_for_alexnet()
    
    print("\n=== AlexNet 학습 추천 클래스 ===")
    for item in good_classes:
        print(f"✓ {item['class']} ({item['korean']})")
        print(f"  카테고리: {item['category']}")
        print(f"  이유: {item['reason']}")
    
    print("\n=== AlexNet 학습 어려운 클래스 ===")
    for item in difficult_classes:
        print(f"✗ {item['class']} ({item['korean']})")
        print(f"  카테고리: {item['category']}")
        print(f"  이유: {item['reason']}")
    
    print("\n=== 추천 사항 ===")
    print("1. 처음 학습: English_springer, church, garbage_truck로 시작")
    print("2. 중급 학습: tench, chain_saw, gas_pump, parachute 추가")
    print("3. 고급 학습: 모든 클래스 포함")
    print("4. 성능 향상: 데이터 증강 (회전, 크기조정, 색상변경) 적용") 