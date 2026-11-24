import sys
from pathlib import Path

import numpy as np
from diffusers.utils import load_image
# pip install insightface==0.7.3
from insightface.app import FaceAnalysis

### 
# https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/165#issue-2055829543
###
class FaceAnalysis2(FaceAnalysis):
    # NOTE: allows setting det_size for each detection call.
    # the model allows it but the wrapping code from insightface
    # doesn't show it, and people end up loading duplicate models
    # for different sizes where there is absolutely no need to
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)

def analyze_faces(face_analysis: FaceAnalysis, img_data: np.ndarray, det_size=(640, 640)):
    # NOTE: try detect faces, if no faces detected, lower det_size until it does
    detection_sizes = [None] + [(size, size) for size in range(640, 256, -64)] + [(256, 256)]

    for size in detection_sizes:
        faces = face_analysis.get(img_data, det_size=size)
        if len(faces) > 0:
            return faces

    return []

if __name__ == "__main__":
    #face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
    face_detector = FaceAnalysis2(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    input_dir = Path(sys.argv[1])
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in image_extensions),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"No input images in '{input_dir}'")

    embeddings = []
    for image_path in image_paths:
        image = np.asarray(load_image(str(image_path)))[:, :, ::-1]
        faces = analyze_faces(face_detector, image)
        if not faces:
            raise ValueError(f"No face detected in '{image_path}'")
        embedding = np.asarray(faces[0]["embedding"], dtype=np.float32)
        if embedding.shape != (512,):
            raise ValueError(f"Expected a 512-value face embedding for '{image_path}'")
        embeddings.append(embedding)

    np.save(input_dir / "id_embeds.npy", np.stack(embeddings), allow_pickle=False)
