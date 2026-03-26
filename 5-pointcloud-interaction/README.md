# Real-Time Point Cloud Interaction

Intel RealSense D435 depth camera를 이용한 실시간 3D 포인트클라우드 시각화 및 저장 시스템.
OpenCV + NumPy 기반의 소프트웨어 렌더러로 하드웨어 가속 없이 3D 포인트클라우드를 렌더링하며, ROS 연동을 통해 로봇 도달 시 자동 캡처를 수행한다.

---

## 파이프라인

```
[Step 1] RealSense Pipeline ── Depth + Color 스트림 캡처 (30 FPS)
         ↓
[Step 2] Depth Filtering ───── Threshold → Decimation → Spatial → Temporal
         ↓
[Step 3] Point Cloud 생성 ──── rs.pointcloud().calculate(depth_frame)
         ↓
[Step 4] View Transform ────── 회전/이동 적용 → 3D → 2D 투영
         ↓
[Step 5] Software Rendering ── Painter's Algorithm으로 깊이 정렬 후 그리기
```

## 3D → 2D Projection

카메라 좌표계의 3D 포인트를 2D 이미지 평면에 투영한다.

```
View Transform:
  v_view = (v - pivot) · R + pivot - translation

Perspective Projection:
  proj_x = v_x / v_z × (w × aspect) + w/2
  proj_y = v_y / v_z × h + h/2

Near Clipping:
  v_z < 0.03 → 제거 (NaN 처리)
```

- `pivot`: 회전 중심점 (translation + [0, 0, distance])
- `R`: pitch/yaw로 구성한 회전 행렬 (Rodrigues)
- near clipping으로 카메라 뒤의 포인트 제거

## Painter's Algorithm

Z-buffer 없이 올바른 깊이 순서로 포인트를 렌더링한다.

```
1. View space에서 각 포인트의 z값 계산
2. z값 기준 내림차순 정렬 (뒤 → 앞)
3. 정렬된 순서대로 2D에 그리기 → 가까운 포인트가 먼 포인트를 덮음

out[i, j] = color[u, v]  (UV mapping으로 텍스처 색상 할당)
```

- 하드웨어 가속 없이도 올바른 occlusion 처리 가능
- Decimation으로 포인트 수를 줄여 실시간 성능 확보

## Depth Filtering

RealSense SDK의 post-processing 필터를 순차 적용하여 깊이 품질을 개선한다.

| 필터 | 역할 | 주요 파라미터 |
|------|------|-------------|
| Threshold | 최대 거리 기반 필터링 (0.8m) | min/max distance |
| Decimation | 해상도 축소 (2^n배) | magnitude: 1~3 |
| Spatial | 공간 평활화 (엣지 보존) | alpha=0.5, delta=20 |
| Temporal | 시간 평활화 (프레임 간 안정화) | alpha=0.2, delta=20 |

## ROS 연동

ROS 토픽을 구독하여 로봇이 목표 위치에 도달하면 자동으로 포인트클라우드를 캡처한다.

```
rospy.Subscriber("robot_status", String, callback)

callback:
  "Target position reached" 메시지 수신
  → Threshold 필터 적용
  → 포인트클라우드 생성
  → PLY + JSON 저장 (밀리미터 단위)
```

---

## 하드웨어 및 의존성

| 항목 | 설명 |
|------|------|
| 카메라 | Intel RealSense D435 (Depth + RGB) |
| 해상도 | Depth: 가변 (Decimation 적용), Color: 기본 해상도 |
| 프레임 레이트 | 30 FPS |
| 출력 형식 | PLY (Open3D), JSON (좌표 리스트) |

```
pip install numpy opencv-python pyrealsense2 open3d
```

ROS 노드 사용 시 추가로 `rospy`가 필요하다 (ROS 환경에서 제공).

---

## 조작법

**마우스**

| 입력 | 동작 |
|------|------|
| 좌클릭 드래그 | pitch/yaw 회전 |
| 우클릭 드래그 | 평행이동 (view space) |
| 휠 / 중클릭 드래그 | 줌 (translation z축) |

**키보드**

| 키 | 동작 |
|----|------|
| `p` | 일시정지 |
| `r` | 시점 초기화 |
| `d` | Decimation 단계 순환 (1→2→4→1) |
| `z` | 포인트 스케일링 토글 |
| `c` | 컬러/깊이맵 전환 |
| `f` | 거리 필터 토글 (0.8m 이하만 표시) |
| `s` | 스크린샷 저장 (PNG) |
| `e` | 포인트클라우드 내보내기 (PLY + JSON, mm 단위) |
| `q` / `ESC` | 종료 |

---

## 3D 렌더링 요소

| 요소 | 설명 |
|------|------|
| Grid | xz 평면에 10×10 격자 |
| Axes | RGB = XYZ 좌표축 (두께로 pivot/원점 구분) |
| Frustum | 카메라 시야각 시각화 (intrinsics 기반) |
| Point Cloud | UV-mapped 컬러 포인트 또는 depth colormap |

## 포인트클라우드 저장

```
'e' 키 입력 시:
  1. 현재 프레임의 depth → pointcloud 계산
  2. 미터 → 밀리미터 변환 (×1000)
  3. Open3D로 PLY 저장 + JSON으로 좌표 리스트 저장
  4. 파일명에 타임스탬프 자동 포함: out_mm_YYYYMMDD_HHMMSS.{ply,json}
```

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `pointcloud_viewer.py` | 단일 파일 통합 뷰어 — AppState, RealSense 파이프라인, 3D 렌더러, 포인트클라우드 저장, ROS 노드 |

> 원본 프로젝트는 `app_state`, `camera`, `renderer`, `saver`, `viewer`, `ros_node` 등으로 모듈 분리되어 있으며, 이 파일은 통합 버전이다.
