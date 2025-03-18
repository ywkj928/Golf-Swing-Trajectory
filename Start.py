from MainV3 import Main

if __name__ == "__main__":
    model_path = "best-seg010.pt"
    video_path = "slowmotion_output.mp4"  # 비디오 파일 경로 설정
    app = Main(model_path, video_path)
    app.run()