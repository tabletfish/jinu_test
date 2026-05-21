# Roboflow Jetson Nano Docker Inference

Roboflow에서 학습한 모델 `my-first-project-ml6kp/2`를 Jetson Nano에서 Docker Inference Server 방식으로 테스트하는 예제입니다.

이 저장소는 Docker 서버 방식만 사용합니다. 직접 실행 방식의 `roboflow_jetson_camera.py`는 사용하지 않습니다.

## API Key

API key는 GitHub에 올리면 안 되므로 Jetson에서 `git pull` 받은 뒤 로컬 파일에만 넣습니다.

`roboflow_jetson_docker_client.py`에서 아래 값을 본인 키로 바꿉니다.

```python
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
```

## 가상환경

Jetson 기본 OpenCV를 쓰기 위해 `--system-site-packages`를 붙여 가상환경을 만듭니다.

```bash
cd ~/jinu_test
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install inference-sdk
```

확인:

```bash
which python
python -c "import cv2; print(cv2.__version__, cv2.__file__)"
python -c "from inference_sdk import InferenceHTTPClient; print('sdk ok')"
```

## Docker 서버 실행

처음 한 번만 이미지를 다운로드하며 시간이 오래 걸릴 수 있습니다.

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
sudo docker logs -f inference-server
```

다음부터는 이미 받은 서버를 다시 켭니다.

```bash
sudo docker start inference-server
```

서버 끄기:

```bash
sudo docker stop inference-server
```

## 클라이언트 실행

```bash
cd ~/jinu_test
source .venv/bin/activate
python roboflow_jetson_docker_client.py --camera 0
```

화면 없이 로보카 제어용으로 돌릴 때:

```bash
python roboflow_jetson_docker_client.py --camera 0 --no-display --interval 0.1
```

## Git 명령어

최신 코드 받아오기:

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
git commit -m "Update Roboflow Jetson Docker client"
```

GitHub에 올리기:

```bash
git push origin main
```
