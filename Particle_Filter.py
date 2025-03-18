import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, process_noise, measurement_noise, particle_range=(0, 640)):
        self.N = num_particles  # 입자의 개수
        self.sigma_process = process_noise  # 프로세스 노이즈
        self.sigma_measurement = measurement_noise  # 측정 노이즈
        self.particle_range = particle_range  # 입자의 범위 (0 ~ 640)
        
        # 초기 입자 및 가중치 설정
        self.particles = np.random.randint(self.particle_range[0], self.particle_range[1], size=(self.N, 2), dtype=np.int32)  # 초기 입자 위치
        self.weights = np.ones(self.N) / self.N  # 균등 가중치

    # 예측 단계
    def predict(self, velocity):
        # velocity는 (dx, dy) 형태로 받음
        noise = np.random.normal(0, self.sigma_process, size=self.particles.shape)
        predicted_particles = self.particles + velocity + noise
        self.particles = np.clip(predicted_particles, self.particle_range[0], self.particle_range[1]).astype(np.int32)  # 0~640 범위 내로 클리핑 후 int32로 변환

    # 업데이트 단계 (측정값 기반 업데이트)
    def update(self, measurement):
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        
        # 가중치 계산: 너무 큰 거리가 있을 경우 가중치가 매우 작거나 NaN이 될 수 있기 때문에 이를 처리
        self.weights = np.exp(-distances**2 / (2 * self.sigma_measurement**2))
        
        # 가중치가 NaN이나 무한대가 포함되면 이를 처리
        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)  # NaN이나 무한대는 0으로 변환
        
        # 가중치 정규화
        weight_sum = np.sum(self.weights)
        
        # 가중치 합이 0일 경우(모든 가중치가 0일 경우) 방지
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # 가중치 합이 0이라면 균등 분포로 초기화
            self.weights = np.ones(self.N) / self.N

    # 재샘플링 단계
    def resample(self):
        # 가중치 배열에 NaN이 있는지 확인
        if np.any(np.isnan(self.weights)):
            print("Warning: NaN values found in weights. Resampling with uniform distribution.")
            self.weights = np.ones(self.N) / self.N  # 균등 가중치로 초기화
        
        # 가중치 배열이 유효한지 확인
        weight_sum = np.sum(self.weights)
        if weight_sum == 0:
            print("Warning: Sum of weights is 0. Resampling with uniform distribution.")
            self.weights = np.ones(self.N) / self.N
        
        # 재샘플링: 가중치에 따라 입자들을 선택
        resampled_particles = self.particles[np.random.choice(self.N, self.N, p=self.weights)]
        self.particles = resampled_particles
        
        # 샘플링된 입자들의 평균 위치를 추정된 위치로 사용
        new_estimate = np.average(self.particles, axis=0, weights=self.weights).astype(np.int32)
        return new_estimate
