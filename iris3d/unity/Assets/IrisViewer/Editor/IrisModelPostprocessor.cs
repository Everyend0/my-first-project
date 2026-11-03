using UnityEditor;

namespace IrisViewer.EditorTools
{
    /// <summary>
    /// Import settings for Models/iris_unity.fbx: generic rig, exact (uncompressed)
    /// joint animation, a looping clip named "IrisBurstLoop", and 32-bit indices
    /// (the mesh has ~78k vertices).
    /// </summary>
    internal sealed class IrisModelPostprocessor : AssetPostprocessor
    {
        public const string ModelFileName = "iris_unity.fbx";
        public const string ClipName = "IrisBurstLoop";

        private bool IsIrisModel =>
            assetPath.Replace('\\', '/').EndsWith("/IrisViewer/Models/" + ModelFileName);

        private void OnPreprocessModel()
        {
            if (!IsIrisModel)
            {
                return;
            }
            var importer = (ModelImporter)assetImporter;
            importer.globalScale = 1f;
            importer.useFileScale = true;
            importer.importBlendShapes = false;
            importer.importCameras = false;
            importer.importLights = false;
            importer.importVisibility = false;
            importer.importNormals = ModelImporterNormals.Import;
            importer.indexFormat = ModelImporterIndexFormat.Auto;
            importer.isReadable = false;
            importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
            importer.animationType = ModelImporterAnimationType.Generic;
            importer.importAnimation = true;
            importer.animationCompression = ModelImporterAnimationCompression.Off;
            importer.optimizeGameObjects = false; // keep Iris_FrontMarker as a transform
            importer.skinWeights = ModelImporterSkinWeights.Standard; // 4 per vertex = what the export uses
        }

        private void OnPreprocessAnimation()
        {
            if (!IsIrisModel)
            {
                return;
            }
            var importer = (ModelImporter)assetImporter;
            ModelImporterClipAnimation[] clips = importer.defaultClipAnimations;
            if (clips == null || clips.Length == 0)
            {
                return;
            }
            // one take: frames 1..31 at 30 fps = exactly 1 second, last pose == first
            ModelImporterClipAnimation clip = clips[0];
            clip.name = ClipName;
            clip.loopTime = true;
            clip.loopPose = false;
            clip.keepOriginalOrientation = true;
            clip.keepOriginalPositionY = true;
            clip.keepOriginalPositionXZ = true;
            importer.clipAnimations = new[] { clip };
        }
    }
}
