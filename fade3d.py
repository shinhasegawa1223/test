import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255))

img_path = 'face.jpg'

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    min_detection_confidence=0.5,
    static_image_mode=True) as face_mesh:

    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    results = face_mesh.process(rgb_image)

    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        for id, lm in enumerate(face_landmarks.landmark):
            print(id, lm.x)
        mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,#mp_face_mesh.FACEMESH_CONTOURS
          landmark_drawing_spec=mark_drawing_spec,
          connection_drawing_spec=mesh_drawing_spec
          )
        cv2.imwrite('result.jpg', annotated_image)
    