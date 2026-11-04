using UnityEngine;

/// Attach to both objects in a suspect pair. Remove after diagnosis.
/// Tick 'showOnScreen' to also list the latest events in the Game view (IMGUI overlay).
public class CollisionDebugger : MonoBehaviour
{
    public bool showOnScreen = false;
    [Tooltip("Top-left corner of the on-screen list, in pixels.")]
    public Vector2 screenPosition = new Vector2(10, 10);

    const int MaxLines = 6;
    readonly System.Collections.Generic.Queue<string> lines = new System.Collections.Generic.Queue<string>();

    void Remember(string line)
    {
        lines.Enqueue($"{Time.time:0.00}s  {line}");
        while (lines.Count > MaxLines) lines.Dequeue();
    }

    void OnGUI()
    {
        if (!showOnScreen) return;
        GUILayout.BeginArea(new Rect(screenPosition.x, screenPosition.y, 420, 40 + 22 * MaxLines), GUI.skin.box);
        GUILayout.Label($"[{name}] collisions");
        if (lines.Count == 0) GUILayout.Label("(none yet)");
        foreach (string l in lines) GUILayout.Label(l);
        GUILayout.EndArea();
    }

    void Start()   // Start, so colliders built in BananaPhysics.Awake are counted
    {
        // compound bodies (e.g. a banana with BananaPhysics) keep their colliders on
        // child objects; count those too
        // only colliders that take part in physics (enabled), grouped by type
        var types = new System.Collections.Generic.SortedDictionary<string, int>();
        int triggers = 0, total = 0;
        foreach (var c in GetComponentsInChildren<Collider>())
        {
            if (!c.enabled) continue;
            string t = c.GetType().Name;
            types[t] = types.TryGetValue(t, out int k) ? k + 1 : 1;
            if (c.isTrigger) triggers++;
            total++;
        }
        var parts = new System.Collections.Generic.List<string>();
        foreach (var kv in types) parts.Add($"{kv.Key} x{kv.Value}");
        var rb  = GetComponent<Rigidbody>();
        Debug.Log(
            $"[Setup] {name} | layer={gameObject.layer} ({LayerMask.LayerToName(gameObject.layer)}) " +
            $"| colliders={total} ({(total > 0 ? string.Join(", ", parts) : "none")}) " +
            $"| triggers={triggers} " +
            $"| rigidbody={(rb != null ? (rb.isKinematic ? "Kinematic" : "Dynamic") : "none")} " +
            $"| active={gameObject.activeInHierarchy}",
            this);
    }

    void OnCollisionEnter(Collision col)
    {
        Debug.Log(
            $"[Collision] {name} hit {col.gameObject.name} | contacts: {col.contactCount} | impulse: {col.impulse.magnitude:F3}",
            this);
        Remember($"hit {col.gameObject.name} ({col.contactCount} contacts, impulse {col.impulse.magnitude:F2})");
    }

    void OnTriggerEnter(Collider other)
    {
        Debug.Log(
            $"[Trigger] {name} entered by {other.gameObject.name} | other layer: {other.gameObject.layer}",
            this);
        Remember($"trigger by {other.gameObject.name}");
    }

    void OnCollisionEnter2D(Collision2D col)
        => Debug.Log(
            $"[2D Collision] {name} hit {col.gameObject.name}. This object is using 2D physics callbacks, not 3D.",
            this);
}
