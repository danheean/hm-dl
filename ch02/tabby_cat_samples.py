# ImageNet n02123045 (tabby cat) 샘플 이미지 시각화
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# ImageNet 클래스 정보
imagenet_class = {
    'id': 'n02123045',
    'index': 281,  # ImageNet 1000 클래스 중 인덱스
    'name': 'tabby, tabby cat',
    'korean': '줄무늬 고양이',
    'description': '특징적인 줄무늬, 점, 선 패턴을 가진 고양이'
}

# Tabby cat 패턴 종류
tabby_patterns = {
    'Mackerel': '세로 줄무늬 (생선뼈 모양)',
    'Classic/Blotched': '소용돌이 무늬',
    'Spotted': '점무늬',
    'Ticked': '털 끝이 다른 색',
    'Rosetted': '장미 모양 점무늬'
}

def show_imagenet_class_info():
    """ImageNet 클래스 정보 출력"""
    print("=" * 50)
    print("ImageNet 클래스 정보")
    print("=" * 50)
    print(f"클래스 ID: {imagenet_class['id']}")
    print(f"클래스 인덱스: {imagenet_class['index']}")
    print(f"영어명: {imagenet_class['name']}")
    print(f"한국어명: {imagenet_class['korean']}")
    print(f"설명: {imagenet_class['description']}")
    
    print("\n" + "=" * 50)
    print("Tabby Cat 패턴 종류")
    print("=" * 50)
    for pattern, description in tabby_patterns.items():
        print(f"• {pattern}: {description}")

def load_sample_image_from_url(url):
    """URL에서 이미지 로드"""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        return None

def create_sample_tabby_visualization():
    """Tabby cat 샘플 시각화"""
    
    # 샘플 이미지 URL들 (Wikimedia Commons에서 공개 라이선스)
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Tabby_Pfaffengrund.JPG/320px-Tabby_Pfaffengrund.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_looking.jpg/320px-Orange_tabby_cat_looking.jpg"
    ]
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'ImageNet {imagenet_class["id"]}: {imagenet_class["korean"]}', fontsize=16, fontweight='bold')
    
    # 첫 번째 이미지 로드 및 표시
    img1 = load_sample_image_from_url(sample_urls[0])
    if img1:
        axes[0, 0].imshow(img1)
        axes[0, 0].set_title('Classic Tabby Pattern\n(고전적인 줄무늬 패턴)')
        axes[0, 0].axis('off')
    
    # 두 번째 이미지 로드 및 표시
    img2 = load_sample_image_from_url(sample_urls[1])
    if img2:
        axes[0, 1].imshow(img2)
        axes[0, 1].set_title('Orange Tabby Cat\n(주황 줄무늬 고양이)')
        axes[0, 1].axis('off')
    
    # 패턴 설명
    axes[1, 0].text(0.1, 0.9, '주요 특징:', fontsize=14, fontweight='bold', transform=axes[1, 0].transAxes)
    features = [
        '• 뚜렷한 줄무늬, 점, 선 패턴',
        '• 이마의 "M" 모양 표시',
        '• 다양한 색상 (검은색, 갈색, 주황색 등)',
        '• 4가지 주요 패턴 타입',
        '• 가장 흔한 고양이 털 색깔'
    ]
    for i, feature in enumerate(features):
        axes[1, 0].text(0.1, 0.8 - i*0.12, feature, fontsize=11, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    # 분류 정보
    axes[1, 1].text(0.1, 0.9, '분류 정보:', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    classification = [
        f'ImageNet ID: {imagenet_class["id"]}',
        f'클래스 인덱스: {imagenet_class["index"]}',
        f'영어명: {imagenet_class["name"]}',
        f'한국어명: {imagenet_class["korean"]}',
        '상위 카테고리: 고양이 (Cat)'
    ]
    for i, info in enumerate(classification):
        axes[1, 1].text(0.1, 0.8 - i*0.12, info, fontsize=11, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_related_classes():
    """관련 ImageNet 클래스들"""
    related_classes = {
        'n02123159': 'tiger cat (호랑이 고양이)',
        'n02123394': 'Persian cat (페르시안 고양이)',
        'n02123597': 'Siamese cat (샴 고양이)',
        'n02124075': 'Egyptian cat (이집트 고양이)',
        'n02125311': 'cougar (쿠거)'
    }
    
    print("\n" + "=" * 50)
    print("관련 ImageNet 고양이 클래스들")
    print("=" * 50)
    for class_id, description in related_classes.items():
        print(f"{class_id}: {description}")

def main():
    """메인 함수"""
    # 클래스 정보 출력
    show_imagenet_class_info()
    
    # 샘플 이미지 시각화
    print("\n샘플 이미지 시각화 중...")
    create_sample_tabby_visualization()
    
    # 관련 클래스 정보
    get_related_classes()
    
    print("\n" + "=" * 50)
    print("💡 추가 정보:")
    print("• Wikimedia Commons에서 더 많은 tabby cat 이미지 확인 가능")
    print("• https://commons.wikimedia.org/wiki/Category:Tabby_cats")
    print("• GitHub imagenet-sample-images 저장소에서 공식 샘플 확인")
    print("=" * 50)

if __name__ == "__main__":
    main() 