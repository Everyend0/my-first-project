using UnityEngine;

/// Attach to both objects in a suspect pair. Remove after diagnosis.
public class CollisionDebugger : MonoBehaviour
{
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
        => Debug.Log(
            $"[Collision] {name} hit {col.gameObject.name} | contacts: {col.contactCount} | impulse: {col.impulse.magnitude:F3}",
            this);

    void OnTriggerEnter(Collider other)
        => Debug.Log(
            $"[Trigger] {name} entered by {other.gameObject.name} | other layer: {other.gameObject.layer}",
            this);

    void OnCollisionEnter2D(Collision2D col)
        => Debug.Log(
            $"[2D Collision] {name} hit {col.gameObject.name}. This object is using 2D physics callbacks, not 3D.",
            this);
}
