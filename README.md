# Golf-Swing-Trajectory 프로젝트

## 개요
본 프로젝트는 OpenCV와 YOLO 모델을 활용하여 골프 헤드의 인식 및 궤적을 그리는 시스템을 개발하는 것을 목표로 합니다. 이 시스템은 실시간 영상 처리 기술을 통해 골프 스윙의 궤적을 정확하게 추적하고 시각화하여, 골프 연습 및 분석에 도움을 주고자 합니다.

## 서론
골프는 정밀한 기술이 요구되는 스포츠로, 스윙의 궤적은 플레이어의 성과에 큰 영향을 미칩니다. 따라서, 골프 스윙의 궤적을 정확히 분석하는 것은 선수의 기술 향상에 필수적입니다. 본 프로젝트에서는 YOLO 모델을 이용한 골프 헤드 탐지와 Bytracker를 통한 추적 기술을 적용하여, 실시간으로 골프 헤드의 궤적을 시각화하는 시스템을 구현하였습니다.

## 본론
프로젝트는 크게 네 가지 모듈로 구성됩니다. 첫째, Main.py에서는 OpenCV를 통해 영상을 호출하고, Object Tracking 및 필터링을 수행하여 골프 헤드의 궤적을 그립니다. 둘째, Object_Tracking.py에서는 YOLO 모델을 사용하여 골프 헤드를 탐지하고, Bytracker를 통해 이를 추적합니다. 셋째, Filter.py에서는 Particle 및 Kalman 필터를 적용하여 영상의 노이즈를 제거하고, 거리 및 속도를 측정합니다. 마지막으로, Tray.py에서는 LSTM 모델을 활용하여 궤적을 예측하고 위치를 추적합니다. 이러한 과정을 통해 실시간 영상에서의 품질 저하와 오검출 문제를 해결하기 위한 추가적인 데이터와 필터링 개선의 필요성을 인식하였습니다.

### 1. 이론적 배경
1. 컴퓨터 비전은 컴퓨터가 이미지나 비디오에서 정보를 추출하고 해석하는 기술입니다. OpenCV(Open Source Computer Vision Library)는 이미지 처리 및 컴퓨터 비전 작업을 위한 라이브러리로, 다양한 기능을 제공합니다. 본 프로젝트에서는 OpenCV를 사용하여 영상 호출, 이미지 후처리, 객체 탐지 및 궤적 시각화를 수행합니다.

2. 객체 탐지는 이미지나 비디오에서 특정 객체를 식별하고 위치를 찾는 과정입니다. 본 프로젝트에서는 YOLO(You Only Look Once) 모델을 사용하여 골프 헤드를 탐지합니다. YOLO는 실시간 객체 탐지에 적합한 딥러닝 기반의 모델로, 이미지를 그리드로 나누고 각 그리드에서 객체를 동시에 예측하는 방식으로 작동합니다. 이로 인해 높은 속도와 정확성을 제공합니다.

3. 객체 추적은 탐지된 객체의 위치를 시간에 따라 지속적으로 추적하는 과정입니다. 본 프로젝트에서는 Bytracker 라이브러리를 활용하여 YOLO로 탐지된 골프 헤드를 추적합니다. Bytracker는 여러 프레임에서 객체의 위치를 추적하기 위해 다양한 알고리즘을 사용하여, 객체의 이동 경로를 안정적으로 추적할 수 있도록 합니다.

4. 필터링 기법 영상 처리에서 노이즈 제거는 중요한 단계입니다. 본 프로젝트에서는 Particle Filter와 Kalman Filter를 사용하여 영상의 노이즈를 제거하고, 골프 헤드의 거리 및 속도를 측정합니다.
Kalman Filter는 선형 동적 시스템에서 상태를 추정하는 알고리즘으로, 예측과 업데이트 단계를 통해 노이즈를 줄이고 정확한 추정을 제공합니다.
Particle Filter는 비선형 및 비가우시안 시스템에서 상태를 추정하는 데 유용하며, 여러 개의 입자를 사용하여 상태 공간을 샘플링합니다.

6. 궤적 예측 LSTM(Long Short-Term Memory) 모델은 시계열 데이터의 패턴을 학습하는 데 적합한 순환 신경망(RNN) 구조입니다. 본 프로젝트에서는 LSTM을 사용하여 골프 헤드의 궤적을 예측합니다. LSTM은 과거의 정보를 기억하고 이를 기반으로 미래의 상태를 예측하는 데 강력한 성능을 보입니다.

### 2. 구현 방법
1. **Main.py**  
   Golf head 인식 및 궤도 그리기
   - OpenCV를 이용한 영상 호출
   - Object_Tracking 호출
   - Filter 호출 및 tray 호출
   - Golf head가 지나간 궤적을 그리기

2. **Object_Tracking.py**  
   YOLO 모델을 이용한 Golf head 탐지 및 Bytracker를 이용한 추적
   - 미리 훈련된 가중치 다운로드 및 로드
   - YOLO fine-tuning
   - Bytracker 라이브러리를 활용한 Tracking 구현 

3. **Filter.py**  
   Particle, Kalman 필터를 사용한 영상 노이즈 제거 및 거리, 속도 측정
   - 이미지 후처리
   - 주어진 영상에서 Golf head 거리 및 속도를 예측

4. **Tray.py**  
   LSTM 모델을 이용한 궤적 예측
   - YOLO 모델을 이용한 좌표를 입력 및 훈련
   - 훈련된 모델을 이용한 궤적 예측 및 위치 추적

     

## 구현 결과


### 결과 이미지
- **정면**:
 <div align="center">
    <img src="images/정면.jpg" style="max-width: 100%; height: auto; width: 400px; height: 280px;">
</div>


- **측면**:
<div align="center">
    <img src="images/측면.jpg" style="max-width: 100%; height: auto; width: 400px; height: 280px;">
</div>


## 결론
검증은 직접 촬영한 비디오는 촬영한 비디오를 사용하여 검증하였습니다. 구현결과에서 볼 수 있듯이 손과 골프채의 헤드 부분이 탐지되어 일정한 궤도를 출력하였습니다. 하지만 실시간 영상에서 품질 저하와 환경변수로 인해 오검출이 발생하였으며, 일부 노이즈가 제거되지 않아 오검출 확률이 높아지는 것을 확인하였습니다. 추가적으로전처리 과정과 필터링을 개선하는 것이 필요하다는 점을 인식했습니다. 또한, 탐지를 위한 추가적인 데이터의 필요성이 나타났습니다. 이번 과제를 통해 저희는 객체 탐지, 추적 예측 모델, 후처리를 위한 필터 제작 코드를 구현하였으며, 이를 기반으로 ML/AI 영상 기반 소프트웨어를 개발하였습니다.


## 참고문헌
1. 이상웅. "허프변환과 YOLO 기반의 골프공 궤적 추적." 한국차세대컴퓨팅학회 논문지 17.2 (2021): 42-52.
2. 이홍로, and 황치정. "화소 및 이동 정보를 이용한 골프 스윙 궤도 추적 알고리즘." 정보처리학회논문지 B 12.5 (2005): 561-566.
3. Zhang, Xiaohan. Golf Ball Detection and Tracking Based on Convolutional Neural Networks. Diss. University of Kansas, 2020.
4. Zhang, Tianxiao, et al. "Efficient golf ball detection and tracking based on convolutional neural networks and kalman filter." arXiv preprint arXiv:2012.09393 (2020).
