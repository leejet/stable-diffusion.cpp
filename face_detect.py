import os
import sys

import numpy as np
import torch
from diffusers.utils import load_image
# pip install insightface==0.7.3
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from safetensors.torch import save_file

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
    #input_folder_name = './scarletthead_woman'
    input_folder_name = sys.argv[1]
    image_basename_list = os.listdir(input_folder_name)
    image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

    input_id_images = []
    for image_path in image_path_list:
        input_id_images.append(load_image(image_path))
    
    id_embed_list = []
    
    for img in input_id_images:
        img = np.array(img)
        img = img[:, :, ::-1]
        faces = analyze_faces(face_detector, img)
        if len(faces) > 0:
            id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))
    
    if len(id_embed_list) == 0:
        raise ValueError(f"No face detected in input image pool")
    
    id_embeds = torch.stack(id_embed_list)    
    
    # for r in id_embeds:
    #     print(r)
    # #torch.save(id_embeds, input_folder_name+'/id_embeds.pt');
    # weights = dict()
    # weights["id_embeds"] = id_embeds
    # save_file(weights, input_folder_name+'/id_embeds.safetensors')

    binary_data = id_embeds.numpy().tobytes()
    two = 4
    zero = 0
    one = 1
    tensor_name = "id_embeds"
# Write binary data to a file
    with open(input_folder_name+'/id_embeds.bin', "wb") as f:
        f.write(two.to_bytes(4, byteorder='little'))
        f.write((len(tensor_name)).to_bytes(4, byteorder='little'))
        f.write(zero.to_bytes(4, byteorder='little'))
        f.write((id_embeds.shape[1]).to_bytes(4, byteorder='little'))
        f.write((id_embeds.shape[0]).to_bytes(4, byteorder='little'))
        f.write(one.to_bytes(4, byteorder='little'))
        f.write(one.to_bytes(4, byteorder='little'))
        f.write(tensor_name.encode('ascii'))
        f.write(binary_data)

    