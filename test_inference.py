# 테스트 이미지 추론 및 시각화
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, regularizers
from keras.preprocessing.image import ImageDataGenerator
import random

def create_regularized_alexnet(num_classes=2, input_shape=(227, 227, 3)):
    """정규화된 AlexNet 모델 (추론용)"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (5, 5), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),
        
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(2048, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.6),
        
        layers.Dense(1024, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.6),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_test_images(test_path, num_samples=12):
    """테스트 이미지 로드"""
    print(f"=== 테스트 이미지 로드 ===")
    
    # 파일 목록 가져오기
    files = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    files.sort()  # 순서대로 정렬
    
    # 랜덤 샘플링
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    print(f"총 테스트 파일 수: {len(files)}")
    print(f"선택된 파일 수: {len(selected_files)}")
    
    images = []
    filenames = []
    
    for filename in selected_files:
        try:
            img_path = os.path.join(test_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((227, 227))
            img_array = np.array(img).astype('float32') / 255.0
            
            images.append(img_array)
            filenames.append(filename)
            
        except Exception as e:
            print(f"에러 ({filename}): {e}")
            continue
    
    return np.array(images), filenames

def visualize_test_images(images, filenames, predictions=None, num_cols=4):
    """테스트 이미지와 예측 결과 시각화"""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        axes[row, col].imshow(images[i])
        
        # 제목 설정
        title = f"{filenames[i]}"
        if predictions is not None:
            pred_class = "Dog" if predictions[i][1] > predictions[i][0] else "Cat"
            confidence = max(predictions[i])
            title += f"\\n{pred_class} ({confidence:.2f})"
            
            # 예측 결과에 따라 색상 설정
            color = 'orange' if pred_class == 'Dog' else 'blue'
            axes[row, col].set_title(title, color=color, fontweight='bold')
        else:
            axes[row, col].set_title(title)
            
        axes[row, col].axis('off')
    
    # 빈 칸 제거
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_simple_model_for_demo():
    """데모용 간단한 모델 (실제 훈련 없이 추론 시연)"""
    model = models.Sequential([
        layers.Input(shape=(227, 227, 3)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    # 랜덤 가중치로 컴파일 (데모용)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def predict_with_analysis(model, images, filenames):
    """예측 및 분석"""
    print("\\n=== 예측 시작 ===")
    
    # 예측 실행
    predictions = model.predict(images, verbose=0)
    
    # 결과 분석
    results = []
    for i, (filename, pred) in enumerate(zip(filenames, predictions)):
        cat_prob = pred[0]
        dog_prob = pred[1]
        
        predicted_class = "Dog" if dog_prob > cat_prob else "Cat"
        confidence = max(cat_prob, dog_prob)
        
        results.append({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'cat_prob': cat_prob,
            'dog_prob': dog_prob
        })
        
        print(f"{filename}: {predicted_class} (신뢰도: {confidence:.3f})")
    
    return predictions, results

def analyze_predictions(results):
    """예측 결과 통계 분석"""
    print("\\n=== 예측 결과 분석 ===")
    
    cat_count = sum(1 for r in results if r['predicted_class'] == 'Cat')
    dog_count = sum(1 for r in results if r['predicted_class'] == 'Dog')
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"Cat 예측: {cat_count}개")
    print(f"Dog 예측: {dog_count}개")
    print(f"평균 신뢰도: {avg_confidence:.3f}")
    
    # 가장 확신하는 예측들
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    print("\\n가장 확신하는 예측 Top 3:")
    for i, result in enumerate(sorted_results[:3]):
        print(f"{i+1}. {result['filename']}: {result['predicted_class']} "
              f"(신뢰도: {result['confidence']:.3f})")
    
    print("\\n가장 불확실한 예측 Top 3:")
    for i, result in enumerate(sorted_results[-3:]):
        print(f"{i+1}. {result['filename']}: {result['predicted_class']} "
              f"(신뢰도: {result['confidence']:.3f})")

def main():
    """메인 함수"""
    # 테스트 데이터 경로
    test_path = 'data/dogs-vs-cats/test'
    
    # 1. 테스트 이미지 로드
    images, filenames = load_test_images(test_path, num_samples=12)
    
    if len(images) == 0:
        print("❌ 테스트 이미지를 찾을 수 없습니다!")
        return
    
    # 2. 이미지 먼저 보여주기
    print("\\n=== 테스트 이미지 시각화 ===")
    visualize_test_images(images, filenames)
    
    # 3. 모델 생성 (데모용)
    print("\\n=== 모델 생성 (데모용) ===")
    model = create_simple_model_for_demo()
    print("⚠️ 주의: 이는 데모용 모델입니다. 실제 성능을 위해서는 완전히 훈련된 모델이 필요합니다.")
    
    # 4. 예측 실행
    predictions, results = predict_with_analysis(model, images, filenames)
    
    # 5. 예측 결과와 함께 이미지 다시 보여주기
    print("\\n=== 예측 결과와 함께 이미지 시각화 ===")
    visualize_test_images(images, filenames, predictions)
    
    # 6. 예측 결과 분석
    analyze_predictions(results)
    
    # 7. 신뢰도별 분포 시각화
    confidences = [r['confidence'] for r in results]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('예측 신뢰도 분포')
    plt.xlabel('신뢰도')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    classes = [r['predicted_class'] for r in results]
    class_counts = [classes.count('Cat'), classes.count('Dog')]
    plt.pie(class_counts, labels=['Cat', 'Dog'], autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'])
    plt.title('예측 클래스 분포')
    
    plt.tight_layout()
    plt.show()
    
    print("\\n=== 추론 완료 ===")
    print("💡 실제 정확한 결과를 위해서는 alexnet_regularized.py로 모델을 완전히 훈련한 후")
    print("   훈련된 모델을 여기서 로드하여 사용하시기 바랍니다.")

if __name__ == "__main__":
    main() 