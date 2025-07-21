# 과적합 해결을 위한 정규화된 AlexNet
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras import layers, models, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random

def create_regularized_alexnet(num_classes=2, input_shape=(227, 227, 3)):
    """과적합 방지를 위한 정규화된 AlexNet"""
    model = models.Sequential([
        # 입력 레이어
        layers.Input(shape=input_shape),
        
        # 첫 번째 컨볼루션 블록
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),  # 추가된 드롭아웃
        
        # 두 번째 컨볼루션 블록
        layers.Conv2D(256, (5, 5), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),
        
        # 세 번째 컨볼루션 블록
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 네 번째 컨볼루션 블록
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 다섯 번째 컨볼루션 블록
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),
        
        # 완전연결 레이어 (더 작게)
        layers.Flatten(),
        layers.Dense(2048, activation='relu',  # 4096 → 2048로 축소
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.6),  # 더 강한 드롭아웃
        
        layers.Dense(1024, activation='relu',  # 4096 → 1024로 축소
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.6),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_data_augmentation():
    """강력한 데이터 증강"""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,      # 더 강한 회전
        width_shift_range=0.3,  # 더 강한 이동
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],  # 밝기 변경
        channel_shift_range=0.2,      # 색상 변경
        fill_mode='nearest'
    )

def improved_data_loader_with_augmentation(data_path, sample_size=4000):
    """데이터 증강을 포함한 개선된 데이터 로더"""
    print(f"=== 데이터 증강 로더 ===")
    
    # 파일 목록 가져오기
    files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    
    # 균형잡힌 샘플링
    cat_files = [f for f in files if f.startswith('cat')]
    dog_files = [f for f in files if f.startswith('dog')]
    
    if sample_size:
        samples_per_class = sample_size // 2
        cat_files = random.sample(cat_files, min(samples_per_class, len(cat_files)))
        dog_files = random.sample(dog_files, min(samples_per_class, len(dog_files)))
    
    selected_files = cat_files + dog_files
    random.shuffle(selected_files)
    
    print(f"선택된 파일 수: {len(selected_files)}")
    
    # 이미지와 라벨 로드
    images = []
    labels = []
    
    for i, filename in enumerate(selected_files):
        try:
            if filename.startswith('cat'):
                label = 0
            elif filename.startswith('dog'):
                label = 1
            else:
                continue
                
            img_path = os.path.join(data_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((227, 227))
            img_array = np.array(img).astype('float32')  # 정규화는 Generator에서
            
            images.append(img_array)
            labels.append(label)
            
            if (i + 1) % 500 == 0:
                print(f"처리 완료: {i + 1}/{len(selected_files)}")
                
        except Exception as e:
            print(f"에러 ({filename}): {e}")
            continue
    
    return np.array(images), np.array(labels)

def train_with_regularization():
    """정규화된 모델 훈련"""
    # 데이터 로드
    data_path = 'data/dogs-vs-cats/train'
    X, y = improved_data_loader_with_augmentation(data_path, sample_size=4000)
    
    # 훈련/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y  # 검증 데이터 비율 증가
    )
    
    # 원-핫 인코딩
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    print(f"훈련 데이터: {X_train.shape}")
    print(f"검증 데이터: {X_val.shape}")
    
    # 데이터 증강 설정
    train_datagen = create_data_augmentation()
    val_datagen = ImageDataGenerator(rescale=1./255)  # 검증용은 증강 없음
    
    # 데이터 제너레이터
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=16,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val_cat,
        batch_size=16,
        shuffle=False
    )
    
    # 정규화된 모델 생성
    model = create_regularized_alexnet(num_classes=2)
    model.summary()
    
    # 모델 컴파일 (더 낮은 학습률)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # 더 낮은 학습률
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 강화된 콜백
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15,  # 더 긴 인내
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3,    # 더 강한 감소
            patience=7, 
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    # 모델 훈련
    print("\n=== 정규화된 모델 훈련 ===")
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def compare_models():
    """기존 모델과 정규화된 모델 비교"""
    model, history = train_with_regularization()
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 손실 그래프
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('정규화된 모델 - Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 정확도 그래프
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('정규화된 모델 - Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 학습률 그래프
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 훈련/검증 차이 그래프
    train_val_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 1].plot(train_val_diff, linewidth=2, color='red')
    axes[1, 1].set_title('Overfitting 지표 (Train - Val Accuracy)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 최종 성능
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n=== 최종 성능 ===")
    print(f"훈련 정확도: {final_train_acc:.4f}")
    print(f"검증 정확도: {final_val_acc:.4f}")
    print(f"과적합 격차: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.1:
        print("✅ 과적합이 잘 제어되었습니다!")
    elif overfitting_gap < 0.15:
        print("⚠️ 약간의 과적합이 있지만 양호합니다.")
    else:
        print("❌ 여전히 과적합 문제가 있습니다.")
    
    return model, history

if __name__ == "__main__":
    model, history = compare_models() 