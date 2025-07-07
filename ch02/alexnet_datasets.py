# AlexNet 학습을 위한 데이터셋 옵션들
import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist
import os

def load_cifar10_tf():
    """CIFAR-10 데이터셋 (TensorFlow) - 32x32 크기, 10 클래스"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 정규화
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 224x224로 리사이즈 (AlexNet 입력 크기)
    x_train = tf.image.resize(x_train, [224, 224])
    x_test = tf.image.resize(x_test, [224, 224])
    
    return (x_train, y_train), (x_test, y_test)

def load_cifar100_tf():
    """CIFAR-100 데이터셋 (TensorFlow) - 32x32 크기, 100 클래스"""
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 224x224로 리사이즈
    x_train = tf.image.resize(x_train, [224, 224])
    x_test = tf.image.resize(x_test, [224, 224])
    
    return (x_train, y_train), (x_test, y_test)

def load_imagenette_torch():
    """Imagenette 데이터셋 (PyTorch) - ImageNet의 서브셋, 10 클래스"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 수동으로 다운로드해야 함
    # wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    # tar -xzf imagenette2-320.tgz
    
    # 데이터 로더 설정
    train_dataset = torchvision.datasets.ImageFolder(
        root='./imagenette2-320/train',
        transform=transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root='./imagenette2-320/val',
        transform=transform
    )
    
    return train_dataset, val_datasetj

def download_kaggle_dataset():
    """Kaggle 데이터셋 다운로드 예시"""
    # kaggle API 설치 필요: pip install kaggle
    # ~/.kaggle/kaggle.json 파일에 API 키 필요
    
    commands = [
        # 개-고양이 데이터셋
        "kaggle competitions download -c dogs-vs-cats",
        
        # 꽃 데이터셋
        "kaggle datasets download -d alxmamaev/flowers-recognition",
        
        # 음식 데이터셋  
        "kaggle datasets download -d kmader/food41"
    ]
    
    return commands

def load_custom_dataset():
    """커스텀 데이터셋 로드 (디렉토리 구조 기반)"""
    # 데이터 디렉토리 구조:
    # data/
    #   train/
    #     class1/
    #       image1.jpg
    #       image2.jpg
    #     class2/
    #       image1.jpg
    #   test/
    #     class1/
    #     class2/
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # 이미지 증강 설정
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 데이터 로더
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, test_generator

# 사용 예시
if __name__ == "__main__":
    # 1. CIFAR-10 (가장 간단)
    print("CIFAR-10 데이터셋 로드...")
    (x_train, y_train), (x_test, y_test) = load_cifar10_tf()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    
    # 2. Imagenette 다운로드 (ImageNet subset)
    print("\n=== Imagenette 데이터셋 다운로드 ===")
    print("다음 명령어를 실행하세요:")
    print("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz")
    print("tar -xzf imagenette2-320.tgz")
    
    # 3. Kaggle 데이터셋 옵션들
    print("\n=== Kaggle 데이터셋 옵션들 ===")
    kaggle_commands = download_kaggle_dataset()
    for cmd in kaggle_commands:
        print(cmd) 