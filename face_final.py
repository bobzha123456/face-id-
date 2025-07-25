import face_recognition
import os 

dic={'geroge.png':'Geroge','me1.jpg':'Chongyun'}
file_name=[]
encs=[]
path = ("/home/czha/Desktop/tiger_folder/face/data")
for lpath in os.listdir(path):
    full_path= os.path.join(path,lpath)
    picture_of_me = face_recognition.load_image_file(full_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
      
    encs.append(my_face_encoding[0])
    file_name.append(lpath)
    print(f"Loaded {len(encs)} known face encodings.")
    if not encs:
        print(f"[Warning] No face found in known image {lpath}, skipping.")
        continue
    

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
path1 = ("/home/czha/Desktop/tiger_folder/face/unknown")
lpath1 = os.listdir(path1)

reverse1 = lpath1[len(lpath1)-1]

full_path1= os.path.join(path1,reverse1)
unknown_picture = face_recognition.load_image_file(full_path1)
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces(encs, unknown_face_encoding)

if any(results):
    
    idx = results.index(True)  
    print(file_name)
    print(idx)
    print(f"{dic[file_name[idx]]}, welcome home!")
else:
    print(f"Unknown person {reverse1},is tring to enter your house")