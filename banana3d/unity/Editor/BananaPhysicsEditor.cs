// BananaPhysicsEditor.cs - IMGUI inspector for BananaPhysics.
// Must live in an "Editor" folder (it uses UnityEditor and is not part of player builds).
//
// Shows what the component found (legs/arms banana, capsules, Rigidbody), warns about the
// things that break these models (colliders on the inside-out Outline_Ink_* shell,
// non-uniform scale), and has Build / Select / Remove buttons. All changes go through Undo.
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(BananaPhysics))]
[CanEditMultipleObjects]
public class BananaPhysicsEditor : Editor
{
    SerializedProperty banana, addRigidbody, mass, collisionDetection, keepUpright, drawGizmos;
    bool showCapsules;

    // cached styles: never allocate GUIStyle inside OnInspectorGUI
    GUIStyle _header;
    GUIStyle Header
    {
        get
        {
            if (_header == null) _header = new GUIStyle(EditorStyles.boldLabel) { fontSize = 13 };
            return _header;
        }
    }

    static readonly GUIContent BananaLabel =
        new GUIContent("Banana", "Auto = recognise legs/arms from the model's child objects.");
    static readonly GUIContent GizmoLabel =
        new GUIContent("Show In Scene", "Draw the collider capsules and the centre of mass when selected.");

    void OnEnable()
    {
        banana = serializedObject.FindProperty("banana");
        addRigidbody = serializedObject.FindProperty("addRigidbody");
        mass = serializedObject.FindProperty("mass");
        collisionDetection = serializedObject.FindProperty("collisionDetection");
        keepUpright = serializedObject.FindProperty("keepUpright");
        drawGizmos = serializedObject.FindProperty("drawGizmos");
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.LabelField("Banana Physics", Header);
        EditorGUILayout.PropertyField(banana, BananaLabel);
        EditorGUILayout.PropertyField(addRigidbody);
        using (new EditorGUI.DisabledGroupScope(!addRigidbody.boolValue))
        {
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(mass);
            EditorGUILayout.PropertyField(collisionDetection);
            EditorGUILayout.PropertyField(keepUpright);
            EditorGUI.indentLevel--;
        }
        EditorGUILayout.PropertyField(drawGizmos, GizmoLabel);

        serializedObject.ApplyModifiedProperties();

        EditorGUILayout.Space();
        if (targets.Length == 1)
            DrawStatus((BananaPhysics)target);
        else
            EditorGUILayout.HelpBox(targets.Length + " bananas selected - the buttons act on all of them.",
                                    MessageType.None);

        EditorGUILayout.Space();
        DrawButtons();
    }

    void DrawStatus(BananaPhysics bp)
    {
        Transform holder = bp.CollidersRoot;
        CapsuleCollider[] caps = holder != null ? holder.GetComponentsInChildren<CapsuleCollider>(false)
                                                : new CapsuleCollider[0];
        Rigidbody rb = bp.GetComponent<Rigidbody>();

        EditorGUILayout.BeginVertical("box");
        EditorGUILayout.LabelField("Detected", bp.DetectedBanana.ToString());
        EditorGUILayout.LabelField("Capsules", caps.Length.ToString());
        EditorGUILayout.LabelField("Rigidbody",
            rb == null ? "none" : (rb.isKinematic ? "Kinematic" : "Dynamic") + ", " + rb.mass.ToString("0.##") + " kg");
        EditorGUILayout.EndVertical();

        if (holder == null)
            EditorGUILayout.HelpBox("No colliders yet. Press Build Colliders (Play mode builds them automatically).",
                                    MessageType.Info);
        int foreign = bp.ForeignColliderCount;
        if (foreign > 0)
            EditorGUILayout.HelpBox(foreign + " other collider(s) under the model, e.g. importer MeshColliders on " +
                                    "Outline_Ink_* (inside-out: they pull objects in). Build Colliders removes " +
                                    "or disables them; also turn off Generate Colliders in the OBJ import settings.",
                                    MessageType.Warning);
        if (bp.HasNonUniformScale)
            EditorGUILayout.HelpBox("Non-uniform scale distorts the capsules. Keep scale x = y = z.",
                                    MessageType.Warning);

        if (caps.Length > 0)
        {
            showCapsules = EditorGUILayout.Foldout(showCapsules, "Capsule list", true);
            if (showCapsules)
            {
                EditorGUI.indentLevel++;
                foreach (CapsuleCollider cc in caps)
                    EditorGUILayout.LabelField(cc.name,
                        "r " + cc.radius.ToString("0.000") + "  h " + cc.height.ToString("0.000"));
                EditorGUI.indentLevel--;
            }
        }
    }

    void DrawButtons()
    {
        bool anyBuilt = false;
        foreach (Object t in targets)
            if (((BananaPhysics)t).CollidersRoot != null) anyBuilt = true;

        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Build Colliders"))
        {
            foreach (Object t in targets) ((BananaPhysics)t).Build();
        }
        using (new EditorGUI.DisabledGroupScope(!anyBuilt))
        {
            if (GUILayout.Button("Select", GUILayout.Width(60)))
            {
                var sel = new System.Collections.Generic.List<Object>();
                foreach (Object t in targets)
                {
                    Transform h = ((BananaPhysics)t).CollidersRoot;
                    if (h != null) sel.Add(h.gameObject);
                }
                Selection.objects = sel.ToArray();
            }
            if (GUILayout.Button("Remove", GUILayout.Width(70)))
            {
                // destroy after this GUI pass, so the inspector layout is not cut mid-draw
                Object[] sel = targets;
                EditorApplication.delayCall += () =>
                {
                    foreach (Object t in sel)
                    {
                        if (t == null) continue;
                        Transform h = ((BananaPhysics)t).CollidersRoot;
                        if (h == null) continue;
                        try
                        {
                            Undo.DestroyObjectImmediate(h.gameObject);
                        }
                        catch (System.InvalidOperationException)
                        {
                            Debug.LogWarning("BananaPhysics: 'Colliders' is part of a prefab - remove it in the " +
                                             "prefab.", t);
                        }
                    }
                };
            }
        }
        EditorGUILayout.EndHorizontal();
    }
}
