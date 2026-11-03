using UnityEngine;
using UnityEngine.EventSystems;

namespace IrisViewer
{
    /// <summary>
    /// Full-screen, invisible UI surface behind the viewer controls.
    /// Dragging on it rotates the mascot; the scroll wheel zooms.
    /// Uses EventSystem events, so it works with both the old Input Manager
    /// and the Input System package.
    /// </summary>
    [RequireComponent(typeof(RectTransform))]
    public sealed class IrisDragSurface : MonoBehaviour, IDragHandler, IScrollHandler
    {
        [SerializeField] private IrisViewerController controller;

        public void OnDrag(PointerEventData eventData)
        {
            if (controller != null)
            {
                controller.RotateBy(eventData.delta);
            }
        }

        public void OnScroll(PointerEventData eventData)
        {
            if (controller != null)
            {
                controller.ZoomBy(eventData.scrollDelta.y);
            }
        }
    }
}
