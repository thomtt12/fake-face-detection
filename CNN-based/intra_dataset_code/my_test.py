# %%
import insightface
import cv2
from pathlib import Path
import json


# model = insightface.model_zoo.get_model('retinaface_r50_v1')
# model.prepare(ctx_id = -1, nms=0.4)


img_dirs = {
    'live': Path('/mnt/c/Users/datnq16/Desktop/Detectedface/ClientFace'),
    'spoof': Path('/mnt/c/Users/datnq16/Desktop/Detectedface/ImposterFace')
}

def detect_face(img_path):
    img = cv2.imread(str(img_path))
    bbox, landmark = model.detect(img, threshold=0.5, scale=0.3)

    a = Path(img_path)
    a.with_name(''.join([a.stem, '_BB.txt'])).write_text(
        ' '.join([str(int(p)) if (p > 1) else str(p) for p in bbox[0]])
    )


label_list = {}
for img_dir in img_dirs:
    for d in img_dirs[img_dir].iterdir():
        img_list = d.glob("*.jpg")
        for f in img_list:
            print(f.name)
            # detect_face(f)
            
            relative_path = str(Path(*f.parts[-3:]))

            label = [0]*44
            if img_dir == 'spoof':
                label[-1] = 1

            label_list[relative_path] = label


    f_name = img_dir +".json"

    with open(img_dirs[img_dir].parents[0]/f_name, 'w') as f:
        json.dump(label_list, f)











# %%
