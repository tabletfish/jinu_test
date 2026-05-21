# Roboflow Jetson Nano Inference Test

Roboflow에서 학습한 모델 `my-first-project-ml6kp/2`를 Jetson Nano 카메라로 테스트하는 예제입니다.

API key는 GitHub에 올리면 안 되므로 코드 실행 전에 `roboflow_jetson_camera.py`의 아래 값을 본인 키로 바꿔서 사용합니다.

```python
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
```

## 1. Direct InferencePipeline 방식

카메라 입력, Roboflow 모델 추론, 화면 표시를 Python 프로세스 하나에서 처리합니다.

```bash
cd /home/dydlz/jinu_test
python3 roboflow_jetson_camera.py --camera 0
```

Jetson Nano에서 느리면 FPS를 낮춰서 실행합니다.

```bash
python3 roboflow_jetson_camera.py --camera 0 --max-fps 5 --confidence 0.5
```

## 2. Docker Inference Server 방식

먼저 Jetson에서 Roboflow Inference Server를 Docker로 실행합니다.

```bash
sudo docker run -d \
  --name inference-server \
  --runtime nvidia \
  --read-only \
  -p 9001:9001 \
  --volume ~/.inference/cache:/tmp:rw \
  --security-opt="no-new-privileges" \
  --cap-drop="ALL" \
  --cap-add="NET_BIND_SERVICE" \
  roboflow/roboflow-inference-server-jetson-4.6.1:latest
```

서버 확인:

```bash
sudo docker ps
```

그 다음 클라이언트를 실행합니다.

```bash
cd /home/dydlz/jinu_test
python3 roboflow_jetson_docker_client.py --camera 0
```

화면 없이 로보카 제어용으로 돌릴 때:

```bash
python3 roboflow_jetson_docker_client.py --camera 0 --no-display --interval 0.1
```

Docker 서버 끄기:

```bash
sudo docker stop inference-server
```

다시 켜기:

```bash
sudo docker start inference-server
```

## Git 명령어

원격 저장소에서 최신 코드 받아오기:

```bash
git pull origin main
```

변경 내용 확인:

```bash
git status
```

변경 파일 추가:

```bash
git add .
```

커밋 만들기:

```bash
git commit -m "Update Roboflow Jetson examples"
```

GitHub에 올리기:

```bash
git push origin main
```
