# 바나나 캐릭터 3D 모델 (banana3d)

손그림 바나나 캐릭터 2개를 **모양 변형 없이** 3D로 만든 결과물입니다.

| 캐릭터 | 원본 그림 | 3D 모델 | 렌더 |
|---|---|---|---|
| `banana_legs` (다리 벌린 바나나) | `src/banana_legs.png` | `output/banana_legs/banana_legs.obj` + `.mtl` | `output/banana_legs/renders/` |
| `banana_arms` (팔 벌린 바나나) | `src/banana_arms.png` | `output/banana_arms/banana_arms.obj` + `.mtl` | `output/banana_arms/renders/` |

## 폴더 구성 (캐릭터마다)

```
output/<이름>/
  <이름>.obj / <이름>.mtl        3D 모델 + 재질 (미터 단위, +Y 위, 캐릭터 정면 = +Z)
  <이름>.blend                    Blender 5.x 장면 (조명·카메라·외곽선 재질 세팅 완료)
  <이름>_turnaround_sheet.png     8방향 한 장 요약
  <이름>_drawing_vs_3d.png        원본 그림 vs 3D 정면 겹침 검사
  <이름>_fidelity.json            겹침 검사 수치
  renders/<이름>_000_front.png               정면
          <이름>_045_front-right-diagonal.png 대각선 (앞-오른쪽)
          <이름>_090_right-side.png           옆 (오른쪽)
          <이름>_135_back-right-diagonal.png  대각선 (뒤-오른쪽)
          <이름>_180_back.png                 뒤
          <이름>_225_back-left-diagonal.png   대각선 (뒤-왼쪽)
          <이름>_270_left-side.png            옆 (왼쪽)
          <이름>_315_front-left-diagonal.png  대각선 (앞-왼쪽)
          <이름>_front_ortho_match.png        원근 없는 정면 (검사용)
```

"오른쪽/왼쪽"은 캐릭터 기준입니다 (90° = 캐릭터의 오른쪽 옆모습).
렌더는 턴테이블 방식이라 카메라와 조명은 고정, 캐릭터가 돕니다.

## OBJ 안의 오브젝트

| 오브젝트 | 재질 | 내용 |
|---|---|---|
| `Banana_Body` | `Banana_Yellow` | 노란 몸통 (그림의 테두리 안쪽 모양 그대로) |
| `Banana_Tips_Black` | `Ink_Black` | 꼭지·발끝·팔끝 검은 부분 (살짝 광택) |
| `Face_Eye_1`, `Face_Eye_2`, `Face_Mouth` | `Ink_Line_Matte` | 얼굴 (그림에서 그대로 따낸 모양, 표면에 살짝 붙은 무광 잉크) |
| (다리 바나나) 앞·뒤 다리 | `Banana_Body` / `Banana_Tips_Black` 안에 포함 | 그림 속 바닥에 닿은 다리를 그대로 복사해 **앞쪽과 뒤쪽**에 하나씩 단 다리입니다. 길이·굵기·휘는 모양·검은 발끝이 원래 다리와 같고, 몸통에서 매끄럽게 이어집니다. 그래서 다리 바나나는 다리가 4방향(왼쪽·오른쪽·앞·뒤)으로 펴져 있습니다. 다리 사이의 노란 삼각형 면은 요청에 따라 비웠습니다 |
| `InkLine_ArmSeam_N` / `_Back` (팔 바나나) | `Ink_Line_Matte` | 팔과 몸 사이 선 (앞·뒤 표면에 그린 선) |
| `InkLine_LegSplit` (다리 바나나) | `Ink_Line_Matte` | 다리 갈라지는 선의 윗부분 |
| `Outline_Ink_*` | `Outline_Ink` | **만화 외곽선 껍데기** (아래 설명) |

### 외곽선(`Outline_Ink_*`) 사용법

그림의 두껍고 구불구불한 검은 테두리는 "인버티드 헐(inverted hull)" 방식으로 들어가 있습니다.
몸보다 선 두께만큼 큰 껍데기를 안팎을 뒤집어 둔 것이라, **뒷면 컬링(back-face culling)이 켜진 곳**에서만 제대로 보입니다.

- Unity / Unreal / 대부분의 웹 뷰어: 기본값이 뒷면 컬링이라 그대로 보입니다.
- Blender에서 OBJ를 직접 불러올 때: `Outline_Ink` 재질 → Settings → *Backface Culling* 체크. (같이 들어 있는 `.blend` 파일은 이미 설정돼 있습니다.)
- 외곽선 없이 쓰고 싶으면 `Outline_Ink_*` 오브젝트만 지우면 됩니다.

- 외곽선을 지우면 발끝이 바닥에서 선 두께(약 3~4 cm)만큼 떠 있게 됩니다. 바닥에 세우려면 모델을 그만큼 내려 주세요.

선 두께는 그림에서 직접 잰 값입니다. 정면·뒷면 실루엣에서는 그림과 똑같은 두께이고,
옆·대각선에서 새로 생기는 테두리는 손으로 그린 선처럼 두께가 조금씩 출렁이게 했습니다.

### 재질(MTL) 메모

- `Kd` 값은 OBJ 관례대로 sRGB(화면 색) 기준입니다. Unity·웹 뷰어에서는 그대로 맞습니다.
- 정점은 소수 6자리로 쓰면서 겹치는 점을 합쳐 두어서, 각 오브젝트가 구멍 없이 닫혀 있습니다.
- Blender의 OBJ 가져오기는 `Kd`를 선형 값으로 읽어 노랑이 살짝 연해집니다. Blender에서는 같이 들어 있는 `.blend`를 쓰거나, Base Color를 (1.0, 0.88, 0.003)으로 바꿔 주세요.
- 법선(`vn`)은 꼭지 경계처럼 각진 곳에서 갈라 두어서, 검정·노랑 경계에 음영 번짐이 없습니다.
- `Banana_Body`는 정점이 65,535개를 넘어서 32비트 인덱스가 필요합니다. Unity(Index Format: Auto)·three.js는 자동으로 처리합니다. 모바일용으로 가볍게 쓰려면 Blender에서 Decimate 하세요.

## Unity 물리 (충돌)

`unity/BananaPhysics.cs` 를 모델 루트에 붙이면 끝입니다. 다리/팔 바나나는 모델 안의 오브젝트 이름(`InkLine_LegSplit`, `InkLine_ArmSeam_*`)으로 알아서 구분합니다.

- 팔·다리마다 캡슐 콜라이더를 만듭니다. 오목한 모양이라 MeshCollider는 움직이는 Rigidbody에서 무시되고, Convex로 하면 다리 사이가 막히기 때문입니다.
- 캡슐 반지름에 검은 외곽선 두께 대부분을 포함해서, 물체가 외곽선에 닿는 순간 부딪힙니다 (선 안으로 파묻히지 않음). 바닥에 세우면 외곽선 아래쪽이 바닥에 닿습니다.
- 다리 바나나의 앞·뒤 다리도 휘어진 모양을 따라 캡슐을 붙입니다.
- `unity/Editor/BananaPhysicsEditor.cs` (IMGUI 인스펙터, 꼭 `Editor` 폴더에): 감지된 바나나·캡슐 수·Rigidbody 상태와 경고를 보여 주고, Build / Select / Remove 버튼이 있습니다. 바나나를 선택하면 씬 뷰에 캡슐과 무게중심이 그려집니다 (`drawGizmos`).
- 임포트 배율과 Unity의 OBJ X축 반전은 `Banana_Body`·`Face_Mouth` 메쉬 크기로 자동 보정합니다. 메쉬 Read/Write 설정은 필요 없습니다.
- 루트에 Rigidbody 하나를 둡니다 (Continuous Speculative). 무게중심은 실제 모델 부피 중심으로 맞춰 둡니다.
- 발바닥이 둥근 바나나라 그냥 두면 실제처럼 넘어집니다. 캐릭터로 세워 두려면 `keepUpright`를 켜세요 (X/Z 회전 고정).
- 에디터에서 다시 만들기(Build Colliders)는 Ctrl+Z로 되돌릴 수 있습니다.
- `Outline_Ink_*` 에는 절대 콜라이더를 달지 마세요. 안팎이 뒤집힌 메쉬라 물체를 안으로 빨아들입니다. 임포트 설정의 *Generate Colliders*도 꺼 두세요. (스크립트가 다른 콜라이더를 지우고, 프리팹 안이라 못 지우면 꺼 둡니다.)
- 충돌 콜백 `OnCollisionEnter(Collision c)` 은 루트 오브젝트(Rigidbody 있는 곳)에서 받습니다. 캡슐은 그 아래 `Colliders` 자식에 있습니다.
- 캡슐 미리보기: `output/<이름>/<이름>_colliders_preview.png`, 수치: `<이름>_colliders.json`
- 디버그용 `unity/CollisionDebugger.cs` 는 바나나 루트와 부딪히는 상대 오브젝트에 붙여 콘솔을 확인한 뒤 지우면 됩니다. `showOnScreen`을 켜면 최근 충돌이 게임 화면에도 표시됩니다.

### 정면 겹침 수치 참고 (다리 바나나)

다리 바나나는 요청에 따라 앞·뒤에 다리를 하나씩 더 달고 다리 사이 노란 면을 비워서, 정면 모습이 원본 그림과 그 부분만 다릅니다.
그래서 `banana_legs_fidelity.json`의 정면 겹침 수치가 그만큼 낮게 나옵니다. 나머지 몸·외곽선·얼굴은 그림 그대로입니다.

## 어떻게 만들었나 (캐릭터 변형 없음)

1. **그림 분석** (`pipeline/segment.py`): 그림을 2배 해상도로 키워 노란 면, 바깥 테두리, 검은 꼭지, 얼굴, 안쪽 선을 서브픽셀 정확도로 분리. 테두리 두께도 위치마다 측정.
2. **부풀리기** (`pipeline/build_mesh.py`): 테두리 안쪽 모양을 그대로 경계로 두고 푸아송 방정식으로 풍선처럼 부풀림. 정면에서 보면 외곽이 그림과 픽셀 단위로 같고, 구불구불한 손그림 떨림도 그대로 남습니다 (스캔 픽셀 계단만 1px 정도로 다듬어서 옆에서 봐도 광택이 매끈합니다). 검은 꼭지·얼굴은 그림의 잉크 경계를 따라 잘라 냈고, 몸 안쪽 선은 표면에 그려 넣었습니다.
3. **장면·렌더** (`pipeline/scene.py`): Blender Cycles, 스튜디오 배경, 키/필/림 조명 → 바닥 그림자, 몸통 음영, 코팅된 장난감 같은 하이라이트 반사.
4. **검사** (`pipeline/compare.py`): 원근 없는 정면 렌더를 원본 그림에 겹쳐서 어긋난 픽셀을 셉니다 (`*_drawing_vs_3d.png`).

다시 만들기: `pip install bpy scipy scikit-image pyamg pillow` 후 `pipeline/run_all.sh`.
