# Face Recognition-Based Identity Verification System

## Overview

This repository implements a simple yet effective face recognition system in Python using the open-source **face_recognition** library (developed by Adam Geitgey), which is built on top of dlib's deep metric learning-based face recognition model. The system encodes facial features from a set of registered ("known") individuals and compares them against test ("unknown") images to perform identity verification.

The primary application demonstrated is a basic access control mechanism:
- Matched faces receive a "welcome home" message.
- Unmatched faces trigger an alert for unknown personnel.

This project serves as an educational example of deploying facial recognition for personal security or authentication tasks, highlighting the ease of use of modern computer vision libraries.

## Prerequisites

- Python 3.6 or higher
- The **face_recognition** library

  Installation:
  ```bash
  pip install face_recognition
  ```

  *Note*: On some systems, additional dependencies (e.g., CMake, build tools, or libopenblas) may be required for dlib compilation. The library works reliably on macOS, Windows, and Linux.

## Repository Contents

- `face_final.py`: Main script performing encoding and comparison.
- Known face images (registered identity):
  - `me1.jpg`
  - `me 1 data.jpg`
- Test images (non-matching examples):
  - `testing not me1.jpg`
  - `testing not me 2.jpg`

## Setup and Directory Configuration

The original script uses hardcoded absolute paths. To make it runnable directly from the cloned repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/bobzha123456/face-id-.git
   cd face-id-
   ```

2. Create dedicated folders for known and unknown images:
   ```bash
   mkdir data unknown
   ```

3. Move the known images to the `data` folder:
   ```bash
   mv "me1.jpg" "me 1 data.jpg" data/
   ```

4. Move the test images to the `unknown` folder:
   ```bash
   mv "testing not me1.jpg" "testing not me 2.jpg" unknown/
   ```

5. Edit `face_final.py` (lines near the top) to use relative paths:
   ```python
   path_known = "data"          # Folder with registered faces
   path_unknown = "unknown"     # Folder with images to verify
   ```

6. (Optional) Update the name mapping dictionary for clearer output:
   ```python
   dic = {
       'me1.jpg': 'Chongyun',
       'me 1 data.jpg': 'Chongyun'   # Same identity
   }
   ```

## Usage

Execute the script from the repository root:

```bash
python3 face_final.py
```

### Expected Behavior
- The script loads and encodes all valid images in the `data` folder.
- For each image in the `unknown` folder:
  - If a match is found (based on facial encoding distance), it prints a welcome message with the registered name.
  - Otherwise, it alerts that an unknown person is detected.

Example output:
```
loading know person：me1.jpg
Chongyun，welcome home! 
unknwon 'testing not me1.jpg' attend to entry！
```

## Customization

- Add more known individuals by placing additional images in the `data` folder and updating the `dic` dictionary.
- Adjust tolerance using `face_recognition.compare_faces(known_encs, unk_enc, tolerance=0.6)` (lower values = stricter matching).
- For real-time webcam verification, extend the script using `cv2` or similar libraries.

## Limitations and Ethical Considerations

- Performance depends on image quality, lighting, angle, and facial expressions.
- The model was trained on diverse datasets but may exhibit biases.
- Facial recognition technology raises significant privacy and ethical concerns; use responsibly and in compliance with local laws.

## References

- face_recognition library: https://github.com/ageitgey/face_recognition
- dlib face recognition model: http://dlib.net/
- Original research: Davis E. King, "Dlib-ml: A Machine Learning Toolkit," Journal of Machine Learning Research, 2009.

This project provides a lightweight, academically oriented introduction to facial recognition on consumer hardware.
