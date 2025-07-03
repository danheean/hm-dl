# 혼공학습단 14기 - 딥러닝 실습 프로젝트

**"혼자 만들면서 공부하는 딥러닝"** 책을 활용한 실습 프로젝트입니다.

## 📖 프로젝트 소개

혼공학습단 14기에 참여하여 딥러닝의 기초부터 실전까지 단계별로 학습하는 프로젝트입니다.
TensorFlow와 Keras를 활용해 다양한 딥러닝 모델을 구현하고 실습합니다.

## 🛠 기술 스택

- **Python** 3.10.16
- **TensorFlow** 2.17.1
- **Keras** 3.5.0
- **TensorFlow Metal** 1.1.0+ (Mac Silicon GPU 가속)
- **NumPy** - 수치 연산
- **Jupyter Notebook** - 실습 환경

## 📂 프로젝트 구조

```
hm-dl/
├── check.ipynb          # TensorFlow/Keras 설치 확인
├── main.py              # 메인 실습 코드
├── pyproject.toml       # 프로젝트 설정
├── uv.lock              # 의존성 관리
└── README.md            # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd hm-dl

# 가상환경 및 의존성 설치 (uv 사용)
uv sync
```

**Mac Silicon 사용자:**

- `tensorflow-metal`이 자동으로 설치되어 GPU 가속을 사용할 수 있습니다
- M1/M2/M3 칩의 Neural Engine을 활용한 딥러닝 가속화 지원

### 2. 설치 확인

```bash
# Jupyter 노트북 실행
jupyter notebook check.ipynb
```

`check.ipynb`를 실행하여 TensorFlow와 Keras가 정상적으로 설치되었는지 확인할 수 있습니다.

### 3. 실습 진행

```bash
# 메인 실습 코드 실행
python main.py
```

## 📝 학습 내용

- [ ] 딥러닝 기초 개념
- [ ] 신경망 구조와 동작 원리
- [ ] 다층 퍼셉트론 (MLP)
- [ ] 합성곱 신경망 (CNN)
- [ ] 순환 신경망 (RNN/LSTM)
- [ ] 실전 프로젝트

## 🔧 개발 환경

- **OS**: macOS (darwin 24.5.0)
- **Shell**: zsh
- **패키지 관리**: uv
- **가상환경**: .venv

## 📋 체크리스트

### 설치 확인

- [x] TensorFlow 2.17.1 설치 완료
- [x] Keras 3.5.0 설치 완료
- [x] 기본 연산 테스트 통과
- [x] 모델 생성/컴파일 테스트 통과
- [ ] tensorflow-metal 설치 (Mac Silicon GPU 가속)
- [ ] Metal GPU 메모리 설정

### 학습 진도

- [ ] 1장: 딥러닝 시작하기
- [ ] 2장: 신경망의 기초
- [ ] 3장: 딥러닝 모델 만들기
- [ ] ...

## 🤝 혼공학습단 14기

함께 공부하는 동료들과 지식을 공유하고 성장하는 프로젝트입니다.

---

**Happy Deep Learning! 🚀**
