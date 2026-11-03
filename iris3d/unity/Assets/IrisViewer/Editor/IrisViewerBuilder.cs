using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering;

namespace IrisViewer.EditorTools
{
    /// <summary>
    /// Tools > Iris Viewer > Build Viewer Scene
    /// Creates Assets/IrisViewer/Scenes/IrisViewer.unity: the animated Iris model,
    /// lights, camera, and a uGUI (Canvas) control overlay:
    ///
    /// IrisViewerCanvas (Screen Space Overlay, 1920x1080 reference, match 0.5)
    /// ├── DragArea        full screen, invisible: drag = rotate, wheel = zoom
    /// ├── TopBar          title (left) + interaction hint (right)
    /// ├── CameraPanel     bottom-left: Front + 15° / Front / Orbit ±20° (ToggleGroup)
    /// └── PlaybackPanel   bottom-centre: Play/Pause, speed label, speed slider, Reset view
    /// </summary>
    public static class IrisViewerBuilder
    {
        private const string Root = "Assets/IrisViewer";
        private const string ModelPath = Root + "/Models/" + IrisModelPostprocessor.ModelFileName;
        private const string MaterialDir = Root + "/Materials";
        private const string AnimDir = Root + "/Animation";
        private const string SceneDir = Root + "/Scenes";
        private const string ScenePath = SceneDir + "/IrisViewer.unity";
        private const float FloorGap = 0.30f; // same gap as the Blender scene

        // the mascot's colours, exactly as in the drawing
        private static readonly Color32 IrisYellow = new Color32(0xFE, 0xDB, 0x02, 0xFF);
        private static readonly Color32 IrisBlack = new Color32(0x00, 0x00, 0x00, 0xFF);
        private static readonly Color32 TextDark = new Color32(0x11, 0x11, 0x11, 0xFF);
        private static readonly Color32 TextMuted = new Color32(0x4A, 0x4A, 0x4A, 0xFF);
        private static readonly Color32 PanelColor = new Color32(0xFA, 0xFA, 0xFA, 0xF2);

        private static Sprite uiSprite;
        private static Sprite backgroundSprite;
        private static Sprite knobSprite;
        private static Sprite checkSprite;

        [MenuItem("Tools/Iris Viewer/Build Viewer Scene")]
        public static void Build()
        {
            var modelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(ModelPath);
            if (modelAsset == null)
            {
                EditorUtility.DisplayDialog("Iris Viewer", "Model not found:\n" + ModelPath, "OK");
                return;
            }
            if (!EnsureTmpEssentials())
            {
                return;
            }
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            {
                return;
            }

            EnsureFolder(MaterialDir);
            EnsureFolder(AnimDir);
            EnsureFolder(SceneDir);
            LoadBuiltinSprites();

            Material yellow = MakeMaterial(MaterialDir + "/Iris_Yellow.mat", IrisYellow, 0.62f);
            Material black = MakeMaterial(MaterialDir + "/Iris_Black.mat", IrisBlack, 0.68f);
            Material floorMat = MakeMaterial(MaterialDir + "/Iris_Floor.mat", Color.white, 0.0f);
            RemapModelMaterials(yellow, black);
            AnimatorController animController = MakeAnimatorController();
            if (animController == null)
            {
                EditorUtility.DisplayDialog("Iris Viewer", "No animation clip found in " + ModelPath, "OK");
                return;
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // ---- lighting: white studio, key light upper-front-left, rim light behind
            RenderSettings.skybox = null;
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.84f, 0.84f, 0.84f);
            Light keyLight = MakeDirectionalLight("KeyLight", 0.55f, LightShadows.Soft);
            keyLight.shadowStrength = 0.6f;
            Light rimLight = MakeDirectionalLight("RimLight", 0.29f, LightShadows.None);

            // ---- camera
            var cameraGo = new GameObject("ViewCamera");
            cameraGo.tag = "MainCamera";
            var viewCamera = cameraGo.AddComponent<Camera>();
            viewCamera.clearFlags = CameraClearFlags.SolidColor;
            viewCamera.backgroundColor = Color.white;
            viewCamera.nearClipPlane = 0.05f;
            viewCamera.farClipPlane = 200f;
            cameraGo.AddComponent<AudioListener>();

            // ---- model
            var model = (GameObject)PrefabUtility.InstantiatePrefab(modelAsset);
            model.name = "Iris";
            var animator = model.GetComponent<Animator>();
            if (animator == null)
            {
                animator = model.AddComponent<Animator>();
            }
            animator.runtimeAnimatorController = animController;
            animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;

            var skinned = model.GetComponentInChildren<SkinnedMeshRenderer>();
            skinned.updateWhenOffscreen = true; // arms reach past the rest bounds
            skinned.shadowCastingMode = ShadowCastingMode.On;
            skinned.sharedMaterials = skinned.sharedMaterials
                .Select(m => m != null && m.name.Contains("Black") ? black : yellow)
                .ToArray();
            Transform frontMarker = FindDeep(model.transform, "Iris_FrontMarker");

            Bounds rest = RestBounds(skinned);
            var pivot = new GameObject("IrisPivot").transform;
            pivot.position = rest.center;
            model.transform.SetParent(pivot, true);

            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            UnityEngine.Object.DestroyImmediate(floor.GetComponent<Collider>());
            floor.transform.position = new Vector3(rest.center.x, rest.min.y - FloorGap, rest.center.z);
            floor.transform.localScale = new Vector3(8f, 1f, 8f);
            var floorRenderer = floor.GetComponent<MeshRenderer>();
            floorRenderer.sharedMaterial = floorMat;
            floorRenderer.shadowCastingMode = ShadowCastingMode.Off;
            floorRenderer.receiveShadows = true;

            // ---- UI
            var ui = BuildCanvas();
            BuildEventSystem();

            // ---- controller wiring
            var viewer = new GameObject("IrisViewer");
            var controller = viewer.AddComponent<IrisViewerController>();
            var so = new SerializedObject(controller);
            SetRef(so, "pivot", pivot);
            SetRef(so, "animator", animator);
            SetRef(so, "modelRenderer", skinned);
            SetRef(so, "frontMarker", frontMarker);
            SetRef(so, "viewCamera", viewCamera);
            SetRef(so, "keyLight", keyLight);
            SetRef(so, "rimLight", rimLight);
            SetRef(so, "playPauseButton", ui.PlayPause);
            SetRef(so, "playPauseLabel", ui.PlayPauseLabel);
            SetRef(so, "speedSlider", ui.Speed);
            SetRef(so, "speedLabel", ui.SpeedLabel);
            SetRef(so, "frontTiltToggle", ui.FrontTilt);
            SetRef(so, "frontToggle", ui.Front);
            SetRef(so, "orbitToggle", ui.Orbit);
            SetRef(so, "resetButton", ui.Reset);
            so.ApplyModifiedPropertiesWithoutUndo();

            var dragSo = new SerializedObject(ui.DragSurface);
            SetRef(dragSo, "controller", controller);
            dragSo.ApplyModifiedPropertiesWithoutUndo();

            controller.ApplyPresetNow(); // aim lights + camera so the scene looks right in edit mode

            EditorSceneManager.SaveScene(scene, ScenePath);
            AddSceneToBuild(ScenePath);
            Selection.activeGameObject = viewer;
            Debug.Log("[Iris Viewer] Built " + ScenePath + ". Press Play to view Iris.");
        }

        // ------------------------------------------------------------------
        // UI
        // ------------------------------------------------------------------
        private struct ViewerUi
        {
            public IrisDragSurface DragSurface;
            public UnityEngine.UI.Button PlayPause;
            public TMP_Text PlayPauseLabel;
            public UnityEngine.UI.Slider Speed;
            public TMP_Text SpeedLabel;
            public UnityEngine.UI.Toggle FrontTilt;
            public UnityEngine.UI.Toggle Front;
            public UnityEngine.UI.Toggle Orbit;
            public UnityEngine.UI.Button Reset;
        }

        private static ViewerUi BuildCanvas()
        {
            var ui = new ViewerUi();
            var canvasGo = new GameObject("IrisViewerCanvas", typeof(RectTransform), typeof(Canvas),
                typeof(UnityEngine.UI.CanvasScaler), typeof(UnityEngine.UI.GraphicRaycaster));
            canvasGo.layer = UiLayer;
            var canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.GetComponent<UnityEngine.UI.CanvasScaler>();
            scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);
            scaler.screenMatchMode = UnityEngine.UI.CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;
            Transform root = canvasGo.transform;

            // DragArea: first child so every control sits on top of it
            RectTransform drag = NewRect("DragArea", root);
            Stretch(drag, Vector2.zero, Vector2.zero);
            var dragImage = drag.gameObject.AddComponent<UnityEngine.UI.Image>();
            dragImage.color = new Color(1f, 1f, 1f, 0f);
            dragImage.raycastTarget = true;
            ui.DragSurface = drag.gameObject.AddComponent<IrisDragSurface>();

            // TopBar
            RectTransform top = NewRect("TopBar", root);
            Place(top, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), Vector2.zero, new Vector2(0f, 104f));
            TextMeshProUGUI title = MakeText("Title", top, "IRIS VIEWER", 44f, FontStyles.Bold, TextDark,
                TextAlignmentOptions.Left);
            title.rectTransform.anchorMin = new Vector2(0f, 0f);
            title.rectTransform.anchorMax = new Vector2(0.5f, 1f);
            title.rectTransform.offsetMin = new Vector2(48f, 0f);
            title.rectTransform.offsetMax = Vector2.zero;
            TextMeshProUGUI hint = MakeText("Hint", top, "Drag: rotate   ·   Scroll: zoom", 26f, FontStyles.Normal,
                TextMuted, TextAlignmentOptions.Right);
            hint.rectTransform.anchorMin = new Vector2(0.5f, 0f);
            hint.rectTransform.anchorMax = new Vector2(1f, 1f);
            hint.rectTransform.offsetMin = Vector2.zero;
            hint.rectTransform.offsetMax = new Vector2(-48f, 0f);

            // CameraPanel (bottom-left)
            RectTransform cameraPanel = MakePanel("CameraPanel", root);
            Place(cameraPanel, Vector2.zero, Vector2.zero, Vector2.zero, new Vector2(40f, 40f), new Vector2(360f, 252f));
            var vertical = cameraPanel.gameObject.AddComponent<UnityEngine.UI.VerticalLayoutGroup>();
            vertical.padding = new RectOffset(24, 24, 18, 18);
            vertical.spacing = 6f;
            vertical.childAlignment = TextAnchor.UpperLeft;
            vertical.childControlWidth = true;
            vertical.childControlHeight = true;
            vertical.childForceExpandWidth = true;
            vertical.childForceExpandHeight = false;
            var group = cameraPanel.gameObject.AddComponent<UnityEngine.UI.ToggleGroup>();
            group.allowSwitchOff = false;
            TextMeshProUGUI header = MakeText("Header", cameraPanel, "CAMERA", 26f, FontStyles.Bold, TextMuted,
                TextAlignmentOptions.Left);
            header.gameObject.AddComponent<UnityEngine.UI.LayoutElement>().preferredHeight = 40f;
            ui.FrontTilt = MakeToggle("FrontTiltToggle", cameraPanel, "Front + 15°", group, true);
            ui.Front = MakeToggle("FrontToggle", cameraPanel, "Front", group, false);
            ui.Orbit = MakeToggle("OrbitToggle", cameraPanel, "Orbit ±20°", group, false);

            // PlaybackPanel (bottom-centre)
            RectTransform playback = MakePanel("PlaybackPanel", root);
            Place(playback, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                new Vector2(0f, 40f), new Vector2(940f, 104f));
            var horizontal = playback.gameObject.AddComponent<UnityEngine.UI.HorizontalLayoutGroup>();
            horizontal.padding = new RectOffset(24, 24, 20, 20);
            horizontal.spacing = 22f;
            horizontal.childAlignment = TextAnchor.MiddleCenter;
            horizontal.childControlWidth = true;
            horizontal.childControlHeight = true;
            horizontal.childForceExpandWidth = false;
            horizontal.childForceExpandHeight = false;
            ui.PlayPause = MakeButton("PlayPauseButton", playback, "Pause", 170f, 64f, out ui.PlayPauseLabel);
            ui.SpeedLabel = MakeText("SpeedLabel", playback, "Speed 1.00x", 28f, FontStyles.Normal, TextDark,
                TextAlignmentOptions.Center);
            var speedLayout = ui.SpeedLabel.gameObject.AddComponent<UnityEngine.UI.LayoutElement>();
            speedLayout.preferredWidth = 190f;
            speedLayout.preferredHeight = 64f;
            ui.Speed = MakeSlider("SpeedSlider", playback, 260f, 40f, 0.25f, 2f, 1f);
            ui.Reset = MakeButton("ResetButton", playback, "Reset view", 190f, 64f, out _);
            return ui;
        }

        private static RectTransform MakePanel(string name, Transform parent)
        {
            RectTransform rt = NewRect(name, parent);
            var image = rt.gameObject.AddComponent<UnityEngine.UI.Image>();
            image.sprite = uiSprite;
            image.type = UnityEngine.UI.Image.Type.Sliced;
            image.color = PanelColor;
            image.raycastTarget = true; // clicks on the panel should not rotate the model
            var outline = rt.gameObject.AddComponent<UnityEngine.UI.Outline>();
            outline.effectColor = new Color(0f, 0f, 0f, 0.18f);
            outline.effectDistance = new Vector2(2f, -2f);
            return rt;
        }

        private static UnityEngine.UI.Button MakeButton(string name, Transform parent, string label, float width,
            float height, out TMP_Text labelText)
        {
            RectTransform rt = NewRect(name, parent);
            var layout = rt.gameObject.AddComponent<UnityEngine.UI.LayoutElement>();
            layout.preferredWidth = width;
            layout.preferredHeight = height;
            var image = rt.gameObject.AddComponent<UnityEngine.UI.Image>();
            image.sprite = uiSprite;
            image.type = UnityEngine.UI.Image.Type.Sliced;
            image.color = IrisYellow;
            var outline = rt.gameObject.AddComponent<UnityEngine.UI.Outline>();
            outline.effectColor = IrisBlack;
            outline.effectDistance = new Vector2(2f, -2f);
            var button = rt.gameObject.AddComponent<UnityEngine.UI.Button>();
            button.targetGraphic = image;
            UnityEngine.UI.ColorBlock colors = button.colors;
            colors.normalColor = Color.white;
            colors.highlightedColor = new Color(0.94f, 0.94f, 0.94f, 1f);
            colors.pressedColor = new Color(0.80f, 0.80f, 0.80f, 1f);
            colors.selectedColor = Color.white;
            colors.disabledColor = new Color(0.78f, 0.78f, 0.78f, 0.5f);
            button.colors = colors;
            TextMeshProUGUI text = MakeText("Label", rt, label, 30f, FontStyles.Bold, IrisBlack,
                TextAlignmentOptions.Center);
            Stretch(text.rectTransform, Vector2.zero, Vector2.zero);
            labelText = text;
            return button;
        }

        private static UnityEngine.UI.Toggle MakeToggle(string name, Transform parent, string label,
            UnityEngine.UI.ToggleGroup group, bool isOn)
        {
            RectTransform rt = NewRect(name, parent);
            rt.gameObject.AddComponent<UnityEngine.UI.LayoutElement>().preferredHeight = 54f;
            var toggle = rt.gameObject.AddComponent<UnityEngine.UI.Toggle>();

            RectTransform box = NewRect("Background", rt);
            Place(box, new Vector2(0f, 0.5f), new Vector2(0f, 0.5f), new Vector2(0f, 0.5f), Vector2.zero,
                new Vector2(36f, 36f));
            var boxImage = box.gameObject.AddComponent<UnityEngine.UI.Image>();
            boxImage.sprite = uiSprite;
            boxImage.type = UnityEngine.UI.Image.Type.Sliced;
            boxImage.color = IrisYellow;
            var boxOutline = box.gameObject.AddComponent<UnityEngine.UI.Outline>();
            boxOutline.effectColor = IrisBlack;
            boxOutline.effectDistance = new Vector2(2f, -2f);

            RectTransform check = NewRect("Checkmark", box);
            Stretch(check, new Vector2(4f, 4f), new Vector2(-4f, -4f));
            var checkImage = check.gameObject.AddComponent<UnityEngine.UI.Image>();
            checkImage.sprite = checkSprite;
            checkImage.color = IrisBlack;
            checkImage.raycastTarget = false;

            TextMeshProUGUI text = MakeText("Label", rt, label, 30f, FontStyles.Normal, TextDark,
                TextAlignmentOptions.Left);
            text.rectTransform.anchorMin = Vector2.zero;
            text.rectTransform.anchorMax = Vector2.one;
            text.rectTransform.offsetMin = new Vector2(52f, 0f);
            text.rectTransform.offsetMax = Vector2.zero;

            toggle.targetGraphic = boxImage;
            toggle.graphic = checkImage;
            toggle.group = group;
            toggle.isOn = isOn;
            return toggle;
        }

        private static UnityEngine.UI.Slider MakeSlider(string name, Transform parent, float width, float height,
            float min, float max, float value)
        {
            RectTransform rt = NewRect(name, parent);
            var layout = rt.gameObject.AddComponent<UnityEngine.UI.LayoutElement>();
            layout.preferredWidth = width;
            layout.preferredHeight = height;

            RectTransform background = NewRect("Background", rt);
            background.anchorMin = new Vector2(0f, 0.3f);
            background.anchorMax = new Vector2(1f, 0.7f);
            background.offsetMin = Vector2.zero;
            background.offsetMax = Vector2.zero;
            var backgroundImage = background.gameObject.AddComponent<UnityEngine.UI.Image>();
            backgroundImage.sprite = backgroundSprite;
            backgroundImage.type = UnityEngine.UI.Image.Type.Sliced;
            backgroundImage.color = new Color32(0xD9, 0xD9, 0xD9, 0xFF);

            RectTransform fillArea = NewRect("Fill Area", rt);
            fillArea.anchorMin = new Vector2(0f, 0.3f);
            fillArea.anchorMax = new Vector2(1f, 0.7f);
            fillArea.offsetMin = new Vector2(5f, 0f);
            fillArea.offsetMax = new Vector2(-15f, 0f);
            RectTransform fill = NewRect("Fill", fillArea);
            fill.sizeDelta = new Vector2(10f, 0f);
            var fillImage = fill.gameObject.AddComponent<UnityEngine.UI.Image>();
            fillImage.sprite = uiSprite;
            fillImage.type = UnityEngine.UI.Image.Type.Sliced;
            fillImage.color = IrisYellow;
            fillImage.raycastTarget = false;

            RectTransform handleArea = NewRect("Handle Slide Area", rt);
            Stretch(handleArea, new Vector2(10f, 0f), new Vector2(-10f, 0f));
            RectTransform handle = NewRect("Handle", handleArea);
            handle.sizeDelta = new Vector2(28f, 0f);
            var handleImage = handle.gameObject.AddComponent<UnityEngine.UI.Image>();
            handleImage.sprite = knobSprite;
            handleImage.color = IrisBlack;

            var slider = rt.gameObject.AddComponent<UnityEngine.UI.Slider>();
            slider.fillRect = fill;
            slider.handleRect = handle;
            slider.targetGraphic = handleImage;
            slider.direction = UnityEngine.UI.Slider.Direction.LeftToRight;
            slider.minValue = min;
            slider.maxValue = max;
            slider.value = value;
            return slider;
        }

        private static TextMeshProUGUI MakeText(string name, Transform parent, string content, float size,
            FontStyles style, Color color, TextAlignmentOptions alignment)
        {
            RectTransform rt = NewRect(name, parent);
            var text = rt.gameObject.AddComponent<TextMeshProUGUI>();
            text.text = content;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = color;
            text.alignment = alignment;
            text.raycastTarget = false;
            return text;
        }

        private static void BuildEventSystem()
        {
            var eventSystem = new GameObject("EventSystem", typeof(EventSystem));
#if ENABLE_INPUT_SYSTEM
            Type inputSystemModule =
                Type.GetType("UnityEngine.InputSystem.UI.InputSystemUIInputModule, Unity.InputSystem");
            if (inputSystemModule != null)
            {
                eventSystem.AddComponent(inputSystemModule);
                return;
            }
#endif
            eventSystem.AddComponent<StandaloneInputModule>();
        }

        // ------------------------------------------------------------------
        // assets
        // ------------------------------------------------------------------
        private static bool EnsureTmpEssentials()
        {
            if (Resources.Load<TMP_Settings>("TMP Settings") != null)
            {
                return true;
            }
            TMP_PackageResourceImporter.ImportResources(true, false, false);
            EditorUtility.DisplayDialog("Iris Viewer",
                "TextMeshPro Essential Resources were imported.\n" +
                "When the import finishes, run Tools > Iris Viewer > Build Viewer Scene again.", "OK");
            return false;
        }

        private static Material MakeMaterial(string path, Color color, float smoothness)
        {
            RenderPipelineAsset pipeline = GraphicsSettings.currentRenderPipeline;
            string pipelineName = pipeline != null ? pipeline.GetType().Name : string.Empty;
            Shader shader = null;
            string colorProperty = "_Color";
            string smoothnessProperty = "_Glossiness";
            if (pipelineName.Contains("Universal"))
            {
                shader = Shader.Find("Universal Render Pipeline/Lit");
                colorProperty = "_BaseColor";
                smoothnessProperty = "_Smoothness";
            }
            else if (pipelineName.Contains("HDRenderPipeline"))
            {
                shader = Shader.Find("HDRP/Lit");
                colorProperty = "_BaseColor";
                smoothnessProperty = "_Smoothness";
            }
            if (shader == null)
            {
                shader = Shader.Find("Standard");
                colorProperty = "_Color";
                smoothnessProperty = "_Glossiness";
            }

            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                material.shader = shader;
            }
            // Unity treats material colours as sRGB, so these are the drawing's exact values
            material.SetColor(colorProperty, color);
            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", color);
            }
            material.SetFloat(smoothnessProperty, smoothness);
            if (material.HasProperty("_Metallic"))
            {
                material.SetFloat("_Metallic", 0f);
            }
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void RemapModelMaterials(Material yellow, Material black)
        {
            var importer = (ModelImporter)AssetImporter.GetAtPath(ModelPath);
            importer.AddRemap(new AssetImporter.SourceAssetIdentifier(typeof(Material), "Iris_Yellow"), yellow);
            importer.AddRemap(new AssetImporter.SourceAssetIdentifier(typeof(Material), "Iris_Black"), black);
            importer.SaveAndReimport();
        }

        private static AnimatorController MakeAnimatorController()
        {
            AnimationClip clip = AssetDatabase.LoadAllAssetsAtPath(ModelPath)
                .OfType<AnimationClip>()
                .FirstOrDefault(c => !c.name.StartsWith("__preview__", StringComparison.Ordinal));
            if (clip == null)
            {
                return null;
            }
            string path = AnimDir + "/IrisBurstLoop.controller";
            if (AssetDatabase.LoadAssetAtPath<AnimatorController>(path) != null)
            {
                AssetDatabase.DeleteAsset(path);
            }
            return AnimatorController.CreateAnimatorControllerAtPathWithClip(path, clip);
        }

        private static void LoadBuiltinSprites()
        {
            uiSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UISprite.psd");
            backgroundSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Background.psd");
            knobSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Knob.psd");
            checkSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Checkmark.psd");
        }

        // ------------------------------------------------------------------
        // helpers
        // ------------------------------------------------------------------
        private static int UiLayer
        {
            get
            {
                int layer = LayerMask.NameToLayer("UI");
                return layer >= 0 ? layer : 5;
            }
        }

        private static RectTransform NewRect(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.layer = UiLayer;
            go.transform.SetParent(parent, false);
            return (RectTransform)go.transform;
        }

        /// <summary>Anchor/pivot first, then position and size.</summary>
        private static void Place(RectTransform rt, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot,
            Vector2 anchoredPosition, Vector2 sizeDelta)
        {
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.pivot = pivot;
            rt.anchoredPosition = anchoredPosition;
            rt.sizeDelta = sizeDelta;
        }

        private static void Stretch(RectTransform rt, Vector2 offsetMin, Vector2 offsetMax)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.offsetMin = offsetMin;
            rt.offsetMax = offsetMax;
        }

        private static Light MakeDirectionalLight(string name, float intensity, LightShadows shadows)
        {
            var go = new GameObject(name);
            var light = go.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = Color.white;
            light.intensity = intensity;
            light.shadows = shadows;
            return light;
        }

        private static void SetRef(SerializedObject so, string property, UnityEngine.Object value)
        {
            SerializedProperty p = so.FindProperty(property);
            if (p == null)
            {
                Debug.LogWarning("[Iris Viewer] Missing serialized field: " + property);
                return;
            }
            p.objectReferenceValue = value;
        }

        private static Transform FindDeep(Transform parent, string name)
        {
            foreach (Transform child in parent)
            {
                if (child.name == name)
                {
                    return child;
                }
                Transform found = FindDeep(child, name);
                if (found != null)
                {
                    return found;
                }
            }
            return null;
        }

        private static Bounds RestBounds(SkinnedMeshRenderer skinned)
        {
            Bounds local = skinned.sharedMesh.bounds;
            Matrix4x4 m = skinned.transform.localToWorldMatrix;
            var world = new Bounds(m.MultiplyPoint3x4(local.min), Vector3.zero);
            for (int i = 0; i < 8; i++)
            {
                Vector3 corner = local.center + Vector3.Scale(local.extents,
                    new Vector3((i & 1) == 0 ? -1f : 1f, (i & 2) == 0 ? -1f : 1f, (i & 4) == 0 ? -1f : 1f));
                world.Encapsulate(m.MultiplyPoint3x4(corner));
            }
            return world;
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path))
            {
                return;
            }
            string parent = Path.GetDirectoryName(path).Replace('\\', '/');
            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(path));
        }

        private static void AddSceneToBuild(string path)
        {
            var scenes = EditorBuildSettings.scenes.ToList();
            if (scenes.Any(s => s.path == path))
            {
                return;
            }
            scenes.Insert(0, new EditorBuildSettingsScene(path, true));
            EditorBuildSettings.scenes = scenes.ToArray();
        }
    }
}
