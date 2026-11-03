# 아이리스 3D (Iris 3D)

태호 오빠의 마스코트 **아이리스**를 원본 그림 그대로 3D로 옮긴 모델과 4K 30fps 애니메이션이에요.

| 원본 | 3D (정면, 정지 포즈) |
|---|---|
| ![original](source/iris.png) | ![3d](renders/still_front_rest.png) |

## 결과물

| 파일 | 내용 |
|---|---|
| `model/iris.obj` + `model/iris.mtl` | 정지 포즈(원본 그림과 같은 모양) 3D 모델. 노멀이 포함된 스무스 메시예요 |
| `model/iris.blend` | 블렌더 원본 씬: 모델, 리그(뼈대), 애니메이션, 조명, 그림자, 카메라 3대 |
| `renders/iris_front_tilt_4k30.mp4` | 정면에서 15° 위 카메라, 3840×2160, 30fps, 3초 |
| `renders/iris_front_4k30.mp4` | 완전 정면(직교 카메라), 3840×2160, 30fps, 3초 |
| `renders/iris_orbit_4k30.mp4` | 카메라가 좌우 ±20° 천천히 회전, 3840×2160, 30fps, 3초 |
| `renders/still_*.png` | 4K 스틸 이미지 |

## 원본 지키기 (변형·색상 변경 금지)

- **실루엣**: 그림의 외곽선을 서브픽셀 단위로 따서 메시의 경계로 그대로 썼어요. 정면에서 보면 원본 그림과 같은 모양이에요.
- **검은 외곽선**: 그림에서 잰 두께(약 18px)를 그대로 써서, 몸통 가장자리에 검은 띠 재질로 넣었어요.
- **눈·입**: 그림의 눈 2개와 웃는 입 모양을 메시 경계로 그대로 넣었어요(튀어나온 스티커가 아니라 표면과 같은 높이예요).
- **색상**: 노랑 `#FEDB02 (254, 219, 2)`, 검정 `#000000`. 둘 다 그림에서 직접 측정했어요.
  - 렌더: 블렌더 컬러 관리를 *Standard*로 두고 조명을 보정해서, 하이라이트가 아닌 노란 면이 원본 색으로 나오게 했어요.
  - MTL: `Kd`에 sRGB 값을 그대로 적었어요(블렌더 기본 내보내기는 선형 값을 써서 색이 진해 보이기 때문이에요).
- **얼굴 보호**: 애니메이션 중에도 얼굴이 있는 마디는 늘어나거나 줄어들지 않는 단단한 마디예요. 얼굴은 모양 그대로 움직이기만 해요.

## 애니메이션 (참고 GIF 기반)

참고 GIF는 1초짜리 루프예요(덩어리 → 가시가 '팍' 뻗음 → 살짝 튕김 → 유지하며 떨림 → 쏙 들어감).
아이리스는 줄기 8개가 각각 마디 5개(허브 쪽 4개는 접히는 마디, 끝 1개는 둥근 끝을 지키는 단단한 마디)로 되어 있어요.

1. **접힘**: 마디마디가 허브 쪽으로 접혀 줄기 길이가 원래의 약 40%까지 줄어요
2. **팍!**: 허브 쪽 마디부터 차례로 펴지면서 원래 길이를 약 5% 넘었다가 튕기듯 돌아와요
3. **꿈틀**: 다 뻗은 동안 마디마디를 따라 물결처럼 흔들려요(GIF에서 가시가 떨리는 부분이에요)
4. **쏙**: 다시 허브 쪽으로 접혀 들어가요

줄기마다 시작 시점이 조금씩 달라서 GIF처럼 자연스러워요. 30프레임(1초)마다 정확히 반복돼요(이음매 없는 루프).

## 조명, 그림자, 반사

- 왼쪽 앞 위에서 오는 키 라이트(부드러운 그림자)와 왼쪽 뒤 위에서 오는 림 라이트를 썼어요
- 하얀 스튜디오 배경(바닥과 둥근 뒷벽)에 그림자만 떨어지게 했어요. 배경은 완전한 흰색이에요
- 표면에 클리어코트를 입혀서 위쪽 소프트박스와 조명의 반사 하이라이트가 보여요

## 다시 만들기

```bash
pip install opencv-python-headless scipy triangle      # 그림 추적용
python3 scripts/trace_iris.py source/iris.png build    # 그림 → 메시 데이터
scripts/blender.sh -b --factory-startup --python scripts/build_iris.py -- \
    --trace build/iris_trace.npz --meta build/iris_trace.json \
    --blend model/iris.blend --obj model/iris.obj       # 씬, 리그, 애니메이션, OBJ/MTL
scripts/blender.sh -b model/iris.blend --python scripts/verify_iris.py -- --out build/verify
python3 scripts/compare_iris.py source/iris.png build/verify   # 원본과 비교
scripts/render_all.sh                                   # 4K 프레임 렌더 + MP4 인코딩
```

`scripts/blender.sh`는 블렌더 내장 파이썬에서 numpy를 쓸 수 있게 해 줘요(blender.org 공식 빌드는 numpy가 이미 들어 있어서 그냥 `blender`로 실행해도 돼요).

## 검증 결과

VERIFY_PLACEHOLDER
