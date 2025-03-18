import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, process_noise, measurement_noise, particle_range=(0, 640)):
        self.N = num_particles
        self.sigma_process = process_noise
        self.sigma_measurement = measurement_noise
        self.particle_range = particle_range
        self.particles = np.random.uniform(self.particle_range[0], self.particle_range[1], size=(self.N, 6))  # (x, y, vx, vy, ax, ay)
        self.weights = np.ones(self.N) / self.N

    def predict(self, dt):
        # 위치, 속도, 가속도를 모두 고려한 예측
        noise_position = np.random.normal(0, self.sigma_process, size=(self.N, 2))
        noise_velocity = np.random.normal(0, self.sigma_process, size=(self.N, 2))
        noise_acceleration = np.random.normal(0, self.sigma_process, size=(self.N, 2))

        self.particles[:, 0:2] += self.particles[:, 2:4] * dt + 0.5 * self.particles[:, 4:6] * dt**2 + noise_position
        self.particles[:, 2:4] += self.particles[:, 4:6] * dt + noise_velocity
        self.particles[:, 4:6] += noise_acceleration  # 가속도 업데이트

        # 범위 제한
        self.particles = np.clip(self.particles, self.particle_range[0], self.particle_range[1])

    def update(self, measurement):
        distances = np.linalg.norm(self.particles[:, 0:2] - measurement, axis=1)
        self.weights = np.exp(-distances**2 / (2 * self.sigma_measurement**2))
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.N) / self.N

    def resample(self):
        indices = np.random.choice(self.N, self.N, p=self.weights)
        resampled_particles = self.particles[indices]
        self.particles = resampled_particles

        # 예측된 위치 계산
        new_estimate = np.average(self.particles[:, 0:2], axis=0, weights=self.weights).astype(np.int32)
        return new_estimate

class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.x = initial_state  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # 초기 추정 오차 공분산 행렬
        self.Q = np.eye(4) * process_noise  # 프로세스 노이즈 공분산 행렬
        self.R = np.eye(2) * measurement_noise  # 측정 노이즈 공분산 행렬
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 위치(x, y)만 추적

    def predict(self, dt):
        # 상태 전이 행렬 (속도까지 예측)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])  # 상태 전이 행렬

        self.x = F @ self.x  # 예측된 상태
        self.P = F @ self.P @ F.T + self.Q  # 예측된 오차 공분산

    def update(self, z):
        # 측정값 (2D 위치 값)
        z = np.array([z[0], z[1]])  # [center_x, center_y]

        # 측정 잉여 계산: z - H * x
        y = z - self.H @ self.x  # 여기서 self.x는 4차원 상태 벡터
        # 칼만 이득 계산
        S = self.H @ self.P @ self.H.T + self.R  # S는 측정 예측 오차 공분산
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 칼만 이득

        # 상태 벡터 업데이트
        self.x = self.x + K @ y
        # 오차 공분산 행렬 업데이트
        self.P = (np.eye(4) - K @ self.H) @ self.P

class HybridFilter:
    def __init__(self, num_particles, process_noise, measurement_noise, initial_state):
        # ParticleFilter와 KalmanFilter 초기화
        self.particle_filter = ParticleFilter(num_particles, process_noise, measurement_noise)
        self.kalman_filter = KalmanFilter(initial_state, process_noise, measurement_noise)
        
    def predict(self, velocity, dt):
        # Particle Filter 예측
        self.particle_filter.predict(velocity, dt)
        # Kalman Filter 예측
        self.kalman_filter.predict(dt)
        
    def update(self, measurement):
        # Particle Filter 업데이트
        self.particle_filter.update(measurement)
        # Kalman Filter 업데이트
        self.kalman_filter.update(measurement)
        
    def resample(self):
        # Particle Filter 재샘플링
        particle_estimate = self.particle_filter.resample()
        # Kalman Filter의 추정값
        kalman_estimate = self.kalman_filter.x[:2]  # 위치(x, y)만 추출
        
        # 하이브리드 추정값 계산 (가중 평균)
        hybrid_estimate = (particle_estimate + kalman_estimate) / 2
        return hybrid_estimate

# 하이브리드 필터 초기화
num_particles = 1000
process_noise = 1.0
measurement_noise = 2.0
initial_state = np.array([320, 240, 0, 0])  # [x, y, vx, vy] 초기 상태

hybrid_filter = HybridFilter(num_particles, process_noise, measurement_noise, initial_state)

# 예측 및 업데이트 반복
dt = 0.1  # 시간 간격
velocity = np.array([70, 70])  # 예를 들어 일정한 속도