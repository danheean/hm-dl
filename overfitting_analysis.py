# 과적합과 훈련/검증 정확도 갭 분석
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def analyze_overfitting_gap():
    """과적합 수준과 정확도 갭의 관계 분석"""
    
    # 갭 크기별 분류
    gap_ranges = {
        "정상 학습": {"range": (0, 0.1), "color": "green", "description": "모델이 잘 일반화됨"},
        "약간 과적합": {"range": (0.1, 0.2), "color": "orange", "description": "조금 더 정규화 필요"},
        "심각한 과적합": {"range": (0.2, 0.5), "color": "red", "description": "강한 정규화 필요"},
        "극심한 과적합": {"range": (0.5, 1.0), "color": "darkred", "description": "모델 재설계 필요"}
    }
    
    print("=== 과적합 수준 판단 기준 ===")
    for level, info in gap_ranges.items():
        min_gap, max_gap = info["range"]
        print(f"{level}: {min_gap*100:.0f}% - {max_gap*100:.0f}% 갭")
        print(f"  상태: {info['description']}")
        print()

def simulate_training_scenarios():
    """다양한 훈련 시나리오 시뮬레이션"""
    
    scenarios = {
        "정상 학습": {
            "train_acc": [0.5, 0.65, 0.75, 0.82, 0.85, 0.87, 0.88, 0.89, 0.89, 0.90],
            "val_acc": [0.5, 0.62, 0.72, 0.78, 0.82, 0.84, 0.85, 0.85, 0.85, 0.86],
            "color": "green"
        },
        "약간 과적합": {
            "train_acc": [0.5, 0.65, 0.75, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98],
            "val_acc": [0.5, 0.62, 0.72, 0.78, 0.80, 0.81, 0.80, 0.79, 0.78, 0.77],
            "color": "orange"
        },
        "심각한 과적합": {
            "train_acc": [0.5, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 1.0],
            "val_acc": [0.5, 0.62, 0.70, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60, 0.58],
            "color": "red"
        }
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (scenario, data) in enumerate(scenarios.items()):
        plt.subplot(1, 3, i+1)
        
        epochs = range(1, len(data["train_acc"]) + 1)
        
        plt.plot(epochs, data["train_acc"], 'o-', label="훈련 정확도", 
                color=data["color"], linewidth=2, markersize=6)
        plt.plot(epochs, data["val_acc"], 's--', label="검증 정확도", 
                color=data["color"], alpha=0.7, linewidth=2, markersize=6)
        
        # 갭 영역 표시
        plt.fill_between(epochs, data["train_acc"], data["val_acc"], 
                        alpha=0.2, color=data["color"])
        
        # 최종 갭 계산
        final_gap = data["train_acc"][-1] - data["val_acc"][-1]
        
        plt.title(f"{scenario}\n최종 갭: {final_gap:.1%}", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("정확도")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.05)
    
    plt.tight_layout()
    plt.show()

def create_gap_analysis_chart():
    """갭 크기별 분석 차트"""
    
    # 실제 데이터 시뮬레이션
    gaps = np.array([0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.8])
    train_accs = np.array([0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0])
    val_accs = train_accs - gaps
    
    # 색상 매핑
    colors = []
    for gap in gaps:
        if gap < 0.1:
            colors.append('green')
        elif gap < 0.2:
            colors.append('orange')
        elif gap < 0.5:
            colors.append('red')
        else:
            colors.append('darkred')
    
    plt.figure(figsize=(12, 8))
    
    # 1. 갭 크기 분포
    plt.subplot(2, 2, 1)
    plt.bar(range(len(gaps)), gaps * 100, color=colors, alpha=0.7)
    plt.title("정확도 갭 크기", fontsize=14, fontweight='bold')
    plt.xlabel("모델 번호")
    plt.ylabel("갭 (%)")
    plt.grid(True, alpha=0.3)
    
    # 기준선 표시
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='경고선 (10%)')
    plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='위험선 (20%)')
    plt.legend()
    
    # 2. 훈련 vs 검증 정확도
    plt.subplot(2, 2, 2)
    plt.scatter(train_accs * 100, val_accs * 100, c=colors, s=100, alpha=0.7)
    
    # 이상적인 선 (y=x)
    perfect_line = np.linspace(50, 100, 100)
    plt.plot(perfect_line, perfect_line, 'k--', alpha=0.5, label='이상적 관계 (y=x)')
    
    plt.title("훈련 vs 검증 정확도", fontsize=14, fontweight='bold')
    plt.xlabel("훈련 정확도 (%)")
    plt.ylabel("검증 정확도 (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 갭과 검증 정확도의 관계
    plt.subplot(2, 2, 3)
    plt.scatter(gaps * 100, val_accs * 100, c=colors, s=100, alpha=0.7)
    plt.title("갭 크기 vs 검증 정확도", fontsize=14, fontweight='bold')
    plt.xlabel("정확도 갭 (%)")
    plt.ylabel("검증 정확도 (%)")
    plt.grid(True, alpha=0.3)
    
    # 추세선
    z = np.polyfit(gaps * 100, val_accs * 100, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 80, 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'추세선 (기울기: {z[0]:.1f})')
    plt.legend()
    
    # 4. 과적합 수준별 분포
    plt.subplot(2, 2, 4)
    categories = ['정상\n(0-10%)', '약간\n(10-20%)', '심각\n(20-50%)', '극심\n(50%+)']
    counts = [
        np.sum(gaps < 0.1),
        np.sum((gaps >= 0.1) & (gaps < 0.2)),
        np.sum((gaps >= 0.2) & (gaps < 0.5)),
        np.sum(gaps >= 0.5)
    ]
    colors_cat = ['green', 'orange', 'red', 'darkred']
    
    plt.pie(counts, labels=categories, colors=colors_cat, autopct='%1.0f%%', startangle=90)
    plt.title("과적합 수준별 분포", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def overfitting_solutions():
    """과적합 해결 방법들"""
    
    solutions = {
        "갭 0-10% (정상)": [
            "✅ 현재 상태 유지",
            "✅ 더 많은 에포크 훈련 가능",
            "✅ 모델 복잡도 약간 증가 가능"
        ],
        "갭 10-20% (약간 과적합)": [
            "⚠️ 조기 종료(Early Stopping) 적용",
            "⚠️ 드롭아웃 비율 증가 (0.5 → 0.6)",
            "⚠️ 데이터 증강 강화",
            "⚠️ 학습률 감소"
        ],
        "갭 20-50% (심각한 과적합)": [
            "🔴 강한 정규화 (L1/L2) 적용",
            "🔴 모델 크기 축소",
            "🔴 더 많은 훈련 데이터 확보",
            "🔴 배치 정규화 추가",
            "🔴 드롭아웃 0.7+ 적용"
        ],
        "갭 50%+ (극심한 과적합)": [
            "💀 모델 아키텍처 재설계",
            "💀 훨씬 많은 데이터 필요",
            "💀 전이 학습 고려",
            "💀 앙상블 방법 적용"
        ]
    }
    
    print("=== 갭 크기별 해결 방법 ===\n")
    
    for level, methods in solutions.items():
        print(f"{level}:")
        for method in methods:
            print(f"  {method}")
        print()

def practical_example():
    """실제 예시로 설명"""
    
    print("=== 실제 사례 분석 ===\n")
    
    cases = [
        {
            "name": "Case 1: 좋은 모델",
            "train_acc": 0.87,
            "val_acc": 0.84,
            "analysis": "갭 3% - 모델이 잘 일반화됨. 더 훈련해도 좋음."
        },
        {
            "name": "Case 2: 당신의 이전 모델",
            "train_acc": 0.90,
            "val_acc": 0.70,
            "analysis": "갭 20% - 심각한 과적합. 정규화 필요!"
        },
        {
            "name": "Case 3: 최악의 경우",
            "train_acc": 1.0,
            "val_acc": 0.50,
            "analysis": "갭 50% - 모델이 훈련 데이터만 암기함. 재설계 필요."
        }
    ]
    
    for case in cases:
        gap = case["train_acc"] - case["val_acc"]
        print(f"{case['name']}")
        print(f"  훈련 정확도: {case['train_acc']:.1%}")
        print(f"  검증 정확도: {case['val_acc']:.1%}")
        print(f"  갭: {gap:.1%}")
        print(f"  분석: {case['analysis']}")
        print()

def main():
    """메인 분석 함수"""
    print("🎯 과적합과 정확도 갭의 관계 분석\n")
    
    # 1. 기본 개념 설명
    analyze_overfitting_gap()
    
    # 2. 훈련 시나리오 시뮬레이션
    print("=== 훈련 시나리오별 비교 ===")
    simulate_training_scenarios()
    
    # 3. 상세 분석 차트
    print("\n=== 상세 분석 차트 ===")
    create_gap_analysis_chart()
    
    # 4. 해결 방법
    overfitting_solutions()
    
    # 5. 실제 예시
    practical_example()
    
    print("📊 결론:")
    print("• 갭이 클수록 과적합이 심함")
    print("• 10% 이하: 정상")
    print("• 10-20%: 주의 필요") 
    print("• 20% 이상: 심각한 문제")
    print("• 해결책: 정규화, 데이터 증강, 모델 축소")

if __name__ == "__main__":
    main() 