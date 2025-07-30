import face_recognition
import os

dic = {'geroge.png':'George', 'me1.jpg':'Chongyun'}

known_encs = []
known_names = []

path_known = "/Users/tigerzha/Documents/py_learning/face_rec.py/data"
# 只保留扩展名为图片的可处理文件
for fname in os.listdir(path_known):
    if fname.startswith('.'):  # 跳过隐藏文件
        continue
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        print(f"[Skip] 非图片文件: {fname}")
        continue

    full = os.path.join(path_known, fname)
    try:
        image = face_recognition.load_image_file(full)
    except Exception as e:
        print(f"[Error] 无法识别图像 {fname}: {e}")
        continue

    encs = face_recognition.face_encodings(image)
    if not encs:
        print(f"[Warning] 未在 {fname} 中识别人脸")
        continue

    known_encs.append(encs[0])
    known_names.append(dic.get(fname, fname))
    print(f"已加载已知人脸：{fname}")

# 处理 unknown 文件夹
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
        print(f"[Error] 无法识别图像 {fname}: {e}")
        continue

    encs_u = face_recognition.face_encodings(img)
    if not encs_u:
        print(f"[Warning] 未在未知图像 {fname} 中识别人脸")
        continue

    unk_enc = encs_u[0]
    results = face_recognition.compare_faces(known_encs, unk_enc)
    distances = face_recognition.face_distance(known_encs, unk_enc)

    if any(results):
        best = distances.argmin()
        print(f"{known_names[best]}，welcome home! 匹配文件：{fname}")
    else:
        print(f"未知人员 '{fname}' 尝试进入！")
