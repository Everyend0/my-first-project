using TMPro;
using UnityEngine;

namespace IrisViewer
{
    public enum IrisCameraMode
    {
        FrontTilt = 0, // front, 15 degrees above (matches iris_front_tilt_4k30.mp4)
        Front = 1,     // straight-on orthographic (matches iris_front_4k30.mp4)
        Orbit = 2,     // sways +-20 degrees over 3 s (matches iris_orbit_4k30.mp4)
    }

    /// <summary>
    /// Iris mascot viewer: three camera presets, play/pause, playback speed,
    /// drag-to-rotate and scroll-to-zoom. All references are assigned by
    /// Tools > Iris Viewer > Build Viewer Scene, or by hand in the Inspector.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IrisViewerController : MonoBehaviour
    {
        [Header("Scene")]
        [Tooltip("Parent of the model, placed at the model's centre. Dragging rotates it.")]
        [SerializeField] private Transform pivot;
        [SerializeField] private Animator animator;
        [Tooltip("SkinnedMeshRenderer of Iris_Body (used to measure the model).")]
        [SerializeField] private SkinnedMeshRenderer modelRenderer;
        [Tooltip("Iris_FrontMarker from the FBX: tells which side the face is on.")]
        [SerializeField] private Transform frontMarker;
        [SerializeField] private Camera viewCamera;
        [SerializeField] private Light keyLight;
        [SerializeField] private Light rimLight;

        [Header("UI")]
        [SerializeField] private UnityEngine.UI.Button playPauseButton;
        [SerializeField] private TMP_Text playPauseLabel;
        [SerializeField] private UnityEngine.UI.Slider speedSlider;
        [SerializeField] private TMP_Text speedLabel;
        [SerializeField] private UnityEngine.UI.Toggle frontTiltToggle;
        [SerializeField] private UnityEngine.UI.Toggle frontToggle;
        [SerializeField] private UnityEngine.UI.Toggle orbitToggle;
        [SerializeField] private UnityEngine.UI.Button resetButton;

        [Header("Camera presets (same as the Blender renders)")]
        [SerializeField] private IrisCameraMode startMode = IrisCameraMode.FrontTilt;
        [SerializeField] private float tiltDegrees = 15f;
        [SerializeField] private float orbitElevationDegrees = 9f;
        [SerializeField] private float orbitAmplitudeDegrees = 20f;
        [SerializeField] private float orbitPeriodSeconds = 3f;
        [Tooltip("Vertical field of view of an 85 mm lens on a 16:9 frame.")]
        [SerializeField] private float verticalFov = 13.6f;
        [Tooltip("Frame height divided by model height (room for the burst overshoot).")]
        [SerializeField] private float framing = 1.42f;

        [Header("Interaction")]
        [SerializeField] private float dragDegreesPerPixel = 0.25f;
        [SerializeField] private float maxPitchDegrees = 60f;
        [SerializeField] private float minZoom = 0.5f;
        [SerializeField] private float maxZoom = 2.5f;

        // key / rim light directions from the Blender scene, expressed as
        // (right, up, towards-viewer) components relative to the mascot
        private static readonly Vector3 KeyDirection = new Vector3(0.370f, -0.616f, -0.696f);
        private static readonly Vector3 RimDirection = new Vector3(0.470f, -0.574f, 0.671f);

        private IrisCameraMode mode;
        private bool playing = true;
        private float speed = 1f;
        private float yaw;
        private float pitch;
        private float zoom = 1f;
        private float orbitTime;
        private Vector3 center;
        private Vector3 front = Vector3.back;
        private Vector3 right = Vector3.right;
        private Vector3 modelSize = Vector3.one;
        private Quaternion pivotBaseRotation = Quaternion.identity;

        public IrisCameraMode Mode => mode;
        /// <summary>Horizontal unit vector from the mascot towards its face side.</summary>
        public Vector3 FrontDirection => front;
        public bool IsPlaying => playing;

        private void Awake()
        {
            if (viewCamera == null)
            {
                viewCamera = Camera.main;
            }
            MeasureModel();
            if (pivot != null)
            {
                pivotBaseRotation = pivot.rotation;
            }
            AlignLights();
        }

        private void OnEnable()
        {
            if (playPauseButton != null) playPauseButton.onClick.AddListener(TogglePlay);
            if (resetButton != null) resetButton.onClick.AddListener(ResetView);
            if (speedSlider != null) speedSlider.onValueChanged.AddListener(SetSpeed);
            if (frontTiltToggle != null) frontTiltToggle.onValueChanged.AddListener(OnFrontTiltToggle);
            if (frontToggle != null) frontToggle.onValueChanged.AddListener(OnFrontToggle);
            if (orbitToggle != null) orbitToggle.onValueChanged.AddListener(OnOrbitToggle);
        }

        private void OnDisable()
        {
            if (playPauseButton != null) playPauseButton.onClick.RemoveListener(TogglePlay);
            if (resetButton != null) resetButton.onClick.RemoveListener(ResetView);
            if (speedSlider != null) speedSlider.onValueChanged.RemoveListener(SetSpeed);
            if (frontTiltToggle != null) frontTiltToggle.onValueChanged.RemoveListener(OnFrontTiltToggle);
            if (frontToggle != null) frontToggle.onValueChanged.RemoveListener(OnFrontToggle);
            if (orbitToggle != null) orbitToggle.onValueChanged.RemoveListener(OnOrbitToggle);
        }

        private void Start()
        {
            if (speedSlider != null)
            {
                speedSlider.SetValueWithoutNotify(speed);
            }
            SetMode(startMode);
            ApplyPlayback();
        }

        private void LateUpdate()
        {
            if (playing && mode == IrisCameraMode.Orbit)
            {
                orbitTime += Time.deltaTime * speed;
            }
            ApplyPivotRotation();
            PlaceCamera();
        }

        // ------------------------------------------------------------------
        // public API (also used by the UI)
        // ------------------------------------------------------------------
        public void SetMode(IrisCameraMode newMode)
        {
            mode = newMode;
            if (frontTiltToggle != null) frontTiltToggle.SetIsOnWithoutNotify(mode == IrisCameraMode.FrontTilt);
            if (frontToggle != null) frontToggle.SetIsOnWithoutNotify(mode == IrisCameraMode.Front);
            if (orbitToggle != null) orbitToggle.SetIsOnWithoutNotify(mode == IrisCameraMode.Orbit);
            PlaceCamera();
        }

        public void TogglePlay()
        {
            playing = !playing;
            ApplyPlayback();
        }

        public void SetSpeed(float value)
        {
            speed = Mathf.Max(0f, value);
            ApplyPlayback();
        }

        public void ResetView()
        {
            yaw = 0f;
            pitch = 0f;
            zoom = 1f;
            orbitTime = 0f;
        }

        /// <summary>
        /// Measures the model, aims the lights and places the camera for the
        /// start preset. Also works in edit mode (used by the scene builder).
        /// </summary>
        public void ApplyPresetNow()
        {
            MeasureModel();
            pivotBaseRotation = pivot != null ? pivot.rotation : Quaternion.identity;
            AlignLights();
            mode = startMode;
            PlaceCamera();
        }

        /// <summary>Rotate the mascot by a pointer drag (pixels).</summary>
        public void RotateBy(Vector2 dragPixels)
        {
            yaw += dragPixels.x * dragDegreesPerPixel;
            pitch = Mathf.Clamp(pitch - dragPixels.y * dragDegreesPerPixel, -maxPitchDegrees, maxPitchDegrees);
        }

        /// <summary>Zoom in (positive) or out (negative) by a scroll delta.</summary>
        public void ZoomBy(float scroll)
        {
            if (Mathf.Approximately(scroll, 0f))
            {
                return;
            }
            // one wheel notch is 1 (legacy input), 6 (InputSystemUIInputModule
            // default) or more: never zoom more than one 1.1x step per event, and
            // scale the small deltas trackpads send so they zoom smoothly
            float notches = Mathf.Clamp(scroll, -1f, 1f);
            zoom = Mathf.Clamp(zoom * Mathf.Pow(1.1f, notches), minZoom, maxZoom);
        }

        // ------------------------------------------------------------------
        private void OnFrontTiltToggle(bool on) { if (on) SetMode(IrisCameraMode.FrontTilt); }
        private void OnFrontToggle(bool on) { if (on) SetMode(IrisCameraMode.Front); }
        private void OnOrbitToggle(bool on) { if (on) SetMode(IrisCameraMode.Orbit); }

        private void ApplyPlayback()
        {
            if (animator != null)
            {
                animator.speed = playing ? speed : 0f;
            }
            if (playPauseLabel != null)
            {
                playPauseLabel.text = playing ? "Pause" : "Play";
            }
            if (speedLabel != null)
            {
                speedLabel.text = $"Speed {speed:0.00}x";
            }
        }

        /// <summary>
        /// Measures the rest-pose model: centre, size, and which horizontal axis
        /// points from the mascot towards the viewer (the side with the face).
        /// The mascot is flat, so its thinnest horizontal extent is the depth axis;
        /// the front marker decides the sign.
        /// </summary>
        private void MeasureModel()
        {
            Bounds world = new Bounds(transform.position, Vector3.zero);
            bool haveBounds = false;
            if (modelRenderer != null && modelRenderer.sharedMesh != null)
            {
                Bounds local = modelRenderer.sharedMesh.bounds;
                Matrix4x4 m = modelRenderer.transform.localToWorldMatrix;
                for (int i = 0; i < 8; i++)
                {
                    Vector3 corner = local.center + Vector3.Scale(local.extents,
                        new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1));
                    Vector3 p = m.MultiplyPoint3x4(corner);
                    if (!haveBounds)
                    {
                        world = new Bounds(p, Vector3.zero);
                        haveBounds = true;
                    }
                    else
                    {
                        world.Encapsulate(p);
                    }
                }
            }
            center = pivot != null ? pivot.position : (haveBounds ? world.center : transform.position);
            if (!haveBounds)
            {
                return;
            }
            modelSize = world.size;
            Vector3 depthAxis = world.size.x < world.size.z ? Vector3.right : Vector3.forward;
            float sign = -1f;
            if (frontMarker != null)
            {
                sign = Mathf.Sign(Vector3.Dot(frontMarker.position - world.center, depthAxis));
                if (sign == 0f) sign = -1f;
            }
            front = depthAxis * sign;
            right = Vector3.Cross(Vector3.up, -front);
        }

        private void AlignLights()
        {
            if (keyLight != null)
            {
                Vector3 d = right * KeyDirection.x + Vector3.up * KeyDirection.y + front * KeyDirection.z;
                keyLight.transform.rotation = Quaternion.LookRotation(d.normalized, Vector3.up);
            }
            if (rimLight != null)
            {
                Vector3 d = right * RimDirection.x + Vector3.up * RimDirection.y + front * RimDirection.z;
                rimLight.transform.rotation = Quaternion.LookRotation(d.normalized, Vector3.up);
            }
        }

        private void ApplyPivotRotation()
        {
            if (pivot == null)
            {
                return;
            }
            Quaternion user = Quaternion.AngleAxis(yaw, Vector3.up) * Quaternion.AngleAxis(pitch, right);
            pivot.rotation = user * pivotBaseRotation;
        }

        private void PlaceCamera()
        {
            if (viewCamera == null)
            {
                return;
            }
            float aspect = viewCamera.aspect > 0f ? viewCamera.aspect : 16f / 9f;
            float width = Mathf.Max(modelSize.x, modelSize.z);
            float frameHeight = Mathf.Max(modelSize.y * framing, width * 1.30f / aspect) / zoom;

            Vector3 dir;
            switch (mode)
            {
                case IrisCameraMode.Front:
                    viewCamera.orthographic = true;
                    viewCamera.orthographicSize = frameHeight * 0.5f;
                    viewCamera.transform.position = center + front * 20f;
                    viewCamera.transform.rotation = Quaternion.LookRotation(-front, Vector3.up);
                    return;
                case IrisCameraMode.Orbit:
                    {
                        float e = orbitElevationDegrees * Mathf.Deg2Rad;
                        float sway = orbitAmplitudeDegrees *
                                     Mathf.Sin(2f * Mathf.PI * orbitTime / Mathf.Max(0.01f, orbitPeriodSeconds));
                        dir = Quaternion.AngleAxis(sway, Vector3.up) *
                              (front * Mathf.Cos(e) + Vector3.up * Mathf.Sin(e));
                        break;
                    }
                default:
                    {
                        float e = tiltDegrees * Mathf.Deg2Rad;
                        dir = front * Mathf.Cos(e) + Vector3.up * Mathf.Sin(e);
                        break;
                    }
            }
            viewCamera.orthographic = false;
            viewCamera.fieldOfView = verticalFov;
            float distance = frameHeight * 0.5f / Mathf.Tan(verticalFov * 0.5f * Mathf.Deg2Rad);
            viewCamera.transform.position = center + dir.normalized * distance;
            viewCamera.transform.rotation = Quaternion.LookRotation(center - viewCamera.transform.position, Vector3.up);
        }
    }
}
