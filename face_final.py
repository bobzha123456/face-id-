import face_recognition
import os

dic = {'geroge.png':'George', 'me1.jpg':'Chongyun'}

known_encs = []
known_names = []

path_known = "/Users/tigerzha/Documents/py_learning/face_rec.py/data"

for fname in os.listdir(path_known):
    if fname.startswith('.'):  
        continue
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        print(f"[Skip] image: {fname}")
        continue

    full = os.path.join(path_known, fname)
    try:
        image = face_recognition.load_image_file(full)
    except Exception as e:
        print(f"[Error] can't read image {fname}: {e}")
        continue

    encs = face_recognition.face_encodings(image)
    if not encs:
        print(f"[Warning]  {fname} has no facil information")
        continue

    known_encs.append(encs[0])
    known_names.append(dic.get(fname, fname))
    print(f"loading: {fname}")


path_unknown = "/Users/tigerzha/Documents/py_learning/face_rec.py/unknown"
for fname in os.listdir(path_unknown):
    if fname.startswith('.'):
        continue
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        continue

    full = os.path.join(path_unknown, fname)
    try:
        img = face_recognition.load_image_file(full)
    except Exception as e:
        print(f"[Error] can not read the image {fname}: {e}")
        continue

    encs_u = face_recognition.face_encodings(img)
    if not encs_u:
        print(f"[Warning] can not read the image {fname} ")
        continue

    unk_enc = encs_u[0]
    results = face_recognition.compare_faces(known_encs, unk_enc)
    distances = face_recognition.face_distance(known_encs, unk_enc)

    if any(results):
        best = distances.argmin()
        print(f"{known_names[best]}，welcome home! ：{fname}")
    else:
        print(f"unknwon person '{fname}' attend to entry！")
