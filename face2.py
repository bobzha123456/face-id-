import face_recognition
import os 
encs=[]
path = ("/home/czha/Desktop/tiger_folder/face/data")
for lpath in os.listdir(path):
    full_path= os.path.join(path,lpath)
    picture_of_me = face_recognition.load_image_file(full_path)
    print(picture_of_me)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    encs.append(my_face_encoding[0])
    print(f"Loaded {len(encs)} known face encodings.")
    

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces(encs, unknown_face_encoding)

if any(results):
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
