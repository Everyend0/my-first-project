# 아이리스 뷰어 (Unity · uGUI)

아이리스를 Unity에서 돌려 보고, 애니메이션을 재생하거나 멈추고, 카메라 3종을 바꿔 볼 수 있는 뷰어예요.
화면은 uGUI(Canvas)로 만들었어요.

## 넣는 방법

1. `unity/Assets/IrisViewer` 폴더를 Unity 프로젝트의 `Assets/` 아래에 그대로 복사해요.
2. TextMeshPro가 필요해요. Unity 2021.3 이상, Unity 6 모두 기본으로 들어 있어요.
3. 메뉴에서 **Tools ▸ Iris Viewer ▸ Build Viewer Scene**을 눌러요.
   - TMP Essential Resources가 없으면 자동으로 가져온 다음 "다시 실행해 주세요" 창이 떠요. 한 번 더 누르면 돼요.
4. `Assets/IrisViewer/Scenes/IrisViewer.unity`가 만들어져요. ▶ Play를 누르면 아이리스가 움직여요.

Built-in과 URP 렌더 파이프라인을 자동으로 알아보고 알맞은 Lit 셰이더로 재질을 만들어요. HDRP는 지원하지 않아요(배경, 주변광, 조명 단위를 Volume으로 따로 설정해야 해서 빌더가 안내 창을 띄우고 멈춰요).
조명은 선형 색 공간(Linear)에 맞춰 보정했어요. 프로젝트 설정은 바꾸지 않아요. Built-in에서 *Lights Use Linear Intensity*가 꺼져 있으면 빛 세기를 거기에 맞게 바꿔서 넣고, 프로젝트가 Gamma 색 공간이면 콘솔에 경고를 남겨요.

## 화면 구성 (uGUI)

```
IrisViewerCanvas   Screen Space Overlay · Canvas Scaler 1920×1080 · Match 0.5
├── DragArea        화면 전체를 덮는 투명 영역 · 드래그하면 회전, 휠을 굴리면 확대/축소
├── TopBar          왼쪽에 "IRIS VIEWER" 제목, 오른쪽에 조작 안내
├── CameraPanel     왼쪽 아래 · Front + 15° / Front / Orbit ±20° (토글 그룹, 글자를 눌러도 바뀌어요)
└── PlaybackPanel   오른쪽 아래 · Pause/Play · Speed 1.00x · 속도 슬라이더(0.25–2배) · Reset view
EventSystem         Input System 패키지가 있으면 InputSystemUIInputModule, 없으면 StandaloneInputModule
```

| 카메라 | 내용 | 같은 각도의 4K 영상 |
|---|---|---|
| Front + 15° | 정면에서 15° 위, 85mm 렌즈 | `renders/iris_front_tilt_4k30.mp4` |
| Front | 완전 정면(직교 카메라), 원본 그림과 같은 모양 | `renders/iris_front_4k30.mp4` |
| Orbit ±20° | 3초에 한 번 좌우 ±20° 흔들림 | `renders/iris_orbit_4k30.mp4` |

## 파일

| 경로 | 내용 |
|---|---|
| `Models/iris_unity.fbx` | 메시, 뼈대, 1초 루프 애니메이션 `IrisBurstLoop`, 얼굴 방향 표시 `Iris_FrontMarker` |
| `Scripts/IrisViewerController.cs` | 카메라 프리셋, 재생/정지, 속도, 회전, 확대/축소 |
| `Scripts/IrisDragSurface.cs` | 드래그/휠 입력을 컨트롤러에 넘겨줘요 (EventSystem 이벤트라서 옛 Input Manager와 Input System 모두 동작해요) |
| `Editor/IrisModelPostprocessor.cs` | FBX 임포트 설정: Generic 리그, 애니메이션 무압축, 루프 켬, 정점 7.8만 개라 32비트 인덱스 |
| `Editor/IrisViewerBuilder.cs` | 씬과 UI를 한 번에 만들어 주는 메뉴 |

## 원본 지키기

- 재질 색은 원본 그림과 똑같이 노랑 `#FEDB02`, 검정 `#000000`이에요.
- 애니메이션은 블렌더와 똑같은 동작이에요. 내보낸 FBX를 다시 읽어 블렌더 원본과 비교했더니 모든 정점의 오차가 0.001mm 미만이었어요.
- Unity는 뼈가 부모의 크기 변화를 항상 물려받아서, 블렌더 리그(마디가 부모 크기를 물려받지 않음)를 그대로 쓰면 줄기가 찌그러져요. 그래서 내보낼 때 모든 마디를 허브 바로 아래에 두고 같은 움직임을 프레임마다 구워 넣었어요(`scripts/export_unity_fbx.py`).
- 얼굴이 있는 마디는 크기가 변하지 않는 단단한 마디라서 Unity에서도 얼굴 모양이 그대로예요.
- 정점 하나에 뼈가 최대 4개까지 영향을 줘요. SkinnedMeshRenderer를 Bone4로 고정해서 품질 설정과 상관없이 블렌더와 똑같이 움직여요.

## 참고

- 이 저장소를 만든 서버에는 Unity 에디터가 없어서 C# 코드를 실제 Unity로 컴파일하거나 실행해 보지는 못했어요. 처음 열 때 콘솔에 오류가 뜨면 알려 주세요.
- UI 글자는 영어예요. TMP 기본 폰트(LiberationSans)에는 한글이 없어서, 한글을 쓰려면 한글 폰트로 TMP Font Asset을 만들어 Label들의 폰트를 바꿔 주세요.
- 블렌더 씬처럼 흰 바닥(`Floor`)과 뒤쪽 흰 벽(`BackWall`)이 그림자를 받아요. 벽은 아이리스를 돌려도 닿지 않게 약 1.55m 뒤에 있어요(블렌더는 0.95m). 그림자가 필요 없으면 둘 다 꺼 주세요.
- 화면 비율이 5:4보다 좁으면 두 패널이 겹칠 수 있어요.
