import numpy as np
from sklearn.linear_model import LinearRegression

class Trajectory:

    def __init__(self, point):
        self.__points = []  # points 의 각 요소는 [2] shape의 numpy 배열이어야 함
        self.__length = 0
        self.__missing_count = 0

        self.__points.append(point)

    # Trajectory 포인트를 리턴하는 함수
    def getPoints(self):
        return self.__points

    # Trajectory 포인트를 추가하는 함수
    def addPoint(self, point):
        self.__points.append(point)
        self.__missing_count = 0

        last_points = self.__points[-2:]
        point_diff = last_points[1] - last_points[0]
        point_diff_mag = np.sqrt(point_diff.dot(point_diff))
        self.__length += point_diff_mag

    def getLength(self):
        return self.__length

    def checkNextPoint(self, point):
        points_length = len(self.__points)

        if points_length >= 3:
            return Trajectory.checkTriplet(self.__points[-2:] + [point])
        elif points_length == 0:
            return True
        elif points_length == 1:
            point_diff = point - self.__points[0]
            point_diff_mag = np.sqrt(point_diff.dot(point_diff))
            return (point_diff_mag > 2.0) and (point_diff_mag < 100.0)
        elif points_length == 2:
            return Trajectory.checkTriplet(self.__points + [point])
        
    # Missing 카운트 올리는 함수 -> 추적 계속 여부 리턴
    def upcountMissing(self, num_predictions=5):
        # 추적이 계속 가능하도록 Missing Count의 조건 변경
        if len(self.__points) < 3:
            # 포인트가 3개 미만이어도 추적을 계속 진행
            self.__missing_count = 0  # 여기에 초기화시켜서 계속 추적할 수 있도록 설정

        # 예측된 포인트 추가
        for _ in range(num_predictions):
            nextPoint = self.predictNextPoint(self.__points)
            if nextPoint is None:
                continue  # None이 반환되면 추적을 멈추고 넘어갑니다.
            self.__points.append(nextPoint)

        return True

    # 예측 위치를 계산하는 함수
    @classmethod
    def predictNextPoint(self, points):
        if len(points) >= 5:
            last5Points = points[-5:]
            velocity = [last5Points[i+1] - last5Points[i] for i in range(4)]
            acceleration = [velocity[i+1] - velocity[i] for i in range(3)]

            nextVelocity = velocity[-1] + np.mean(acceleration, axis=0)
            nextPoint = last5Points[-1] + nextVelocity

            return nextPoint
        elif len(points) >= 3:
            last3Points = points[-3:]

            velocity = [last3Points[1] - last3Points[0], last3Points[2] - last3Points[1]]
            acceleration = velocity[1] - velocity[0]

            nextVelocity = velocity[1] + acceleration
            nextPoint = last3Points[2] + nextVelocity

            return nextPoint

        # 예측할 수 없는 경우에는 None 반환
        return None

    # Triplet 조건을 만족하는지 확인하는 함수
    @classmethod
    def checkTriplet(self, points):
        if len(points) != 3:
            return False

        # None 값이 있는지 체크
        if any(p is None for p in points):
            return False

        velocity = [points[1] - points[0], points[2] - points[1]]
        acceleration = velocity[1] - velocity[0]

        velocity_mag = [np.sqrt(velocity[0].dot(velocity[0])), np.sqrt(velocity[1].dot(velocity[1]))]
        if velocity_mag[0] > velocity_mag[1]:
            if velocity_mag[1] / velocity_mag[0] < 0.8:
                return False
        else:
            if velocity_mag[0] / velocity_mag[1] < 0.8:
                return False

        if velocity_mag[0] < 2.0 or velocity_mag[0] > 70.0:
            return False
        if velocity_mag[1] < 2.0 or velocity_mag[1] > 70.0:
            return False

        velocity_dot = velocity[1].dot(velocity[0])
        acceleration_angle = np.arccos(velocity_dot / (velocity_mag[0] * velocity_mag[1]))
        if acceleration_angle > np.deg2rad(90.0):
            return False

        acceleration_mag = np.sqrt(acceleration.dot(acceleration))
        if acceleration_mag > 80.0:
            return False

        if acceleration[0] < -2.0:
            return False

        return True

class TrajectoryManager:

    def __init__(self):
        self.__trajectorys = []

    def getTrajectorys(self):
        return self.__trajectorys

    def setPointsFrame(self, points):
        max_trajectory = (0, 0)
        trajectorys_updated = [False] * len(self.__trajectorys)

        for index, point in enumerate(points):
            isAddedTrajectory = False

            # 기존 Trajectory에 추가되는 포인트인지 확인
            for index, updated in enumerate(trajectorys_updated):
                if updated == False:
                    if self.__trajectorys[index].checkNextPoint(point):
                        self.__trajectorys[index].addPoint(point)
                        trajectorys_updated[index] = True
                        isAddedTrajectory = True

                        trajectory_length = self.__trajectorys[index].getLength()
                        if trajectory_length > max_trajectory[0]:
                            max_trajectory = (trajectory_length, index)

                        break

            # Trajectory에 추가되지 않은 포인트는 신규 Trajectory로 생성
            if isAddedTrajectory == False:
                trajectory_new = Trajectory(point)
                self.__trajectorys.append(trajectory_new)

        # 높은 가능성의 Trajectory가 찾아지면 해당 Trajectory만 남김
        if max_trajectory[0] > 30.0:
            self.__trajectorys = [self.__trajectorys[max_trajectory[1]]]
        else:
            # 업데이트 되지 않은 Trajectory의 Missing Count 증가
            for index, updated in reversed(list(enumerate(trajectorys_updated))):
                if updated == False:
                    if self.__trajectorys[index].upcountMissing() == False:
                        self.__trajectorys.remove(self.__trajectorys[index])

