import cv2
import os
import numpy as np
import mediapipe as mp

# =====================================
# Mediapipe Setup
# =====================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return image, results


def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )


def extract_keypoints(results):
    """
    Extracts world landmarks (x, y, z) from both hands.
    Always returns a fixed-size array of 126 values:
      - Left hand:  21 landmarks × 3 = 63 values  (indices 0..62)
      - Right hand: 21 landmarks × 3 = 63 values  (indices 63..125)
    If a hand is not detected, its 63 values are padded with zeros.
    Hand identity is determined by multi_handedness so Left/Right
    order is always consistent regardless of detection order.
    """
    left  = np.zeros(63)
    right = np.zeros(63)

    # FIX: تأكد أن الطولين متساويان قبل المعالجة
    if (results.multi_hand_world_landmarks and
            results.multi_handedness and
            len(results.multi_hand_world_landmarks) == len(results.multi_handedness)):

        for hand_world_lms, handedness in zip(
            results.multi_hand_world_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"

            coords = []
            for lm in hand_world_lms.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords)

            if label == "Left":
                left = coords
            else:
                right = coords

    return np.concatenate([left, right])   # shape: (126,)


# =====================================
# Image Enhancement
# =====================================
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced


# =====================================
# FIX: التحقق من اكتمال الـ sequence
# =====================================
def get_next_sequence_index(class_id, seq_length):
    """
    يعيد أول رقم sequence غير مكتمل أو جديد.
    يتجاهل الـ sequences الجزئية (عدد frames أقل من seq_length).
    """
    folder = os.path.join(DATA_PATH, str(class_id))
    os.makedirs(folder, exist_ok=True)

    valid_sequences = []
    incomplete_sequences = []

    for f in os.listdir(folder):
        if f.isdigit():
            seq_path = os.path.join(folder, f)
            if os.path.isdir(seq_path):
                npy_count = len([x for x in os.listdir(seq_path) if x.endswith('.npy')])
                if npy_count >= seq_length:
                    valid_sequences.append(int(f))
                else:
                    incomplete_sequences.append((int(f), npy_count))

    # حذف الـ sequences الجزئية لتجنب تلويث بيانات التدريب
    for seq_id, count in incomplete_sequences:
        seq_path = os.path.join(folder, str(seq_id))
        print(f"  ⚠️  Delete partial sequence: class={class_id}, seq={seq_id} ({count}/{seq_length} frames)")
        for npy_file in os.listdir(seq_path):
            os.remove(os.path.join(seq_path, npy_file))
        os.rmdir(seq_path)

    return max(valid_sequences) + 1 if valid_sequences else 0


# =====================================
#   EDIT HERE — Define your signs
# =====================================
BASE_CLASS_ID = 1

signs = [
    'home',
    'living_room',
    'light',
    'table',
    'chair',
    'television',
    'window',
    'phone',
    'bedroom',
    'bed',
    'closet',
    'clothes',
    'kitchen',
    'plate',
    'spoon',
    'fridge',
    'cup',
    'bathroom',
    'toothbrush',
    'toothpaste',
    'shower',
    'sink',
]

display_names = {s: s.upper() for s in signs}
actions = np.arange(BASE_CLASS_ID, BASE_CLASS_ID + len(signs))

# =====================================
# Paths & Settings
# =====================================
DATA_PATH  = "mp_data"
VIDEO_PATH = "vid_data"

TARGET_FPS            = 30
SECONDS_PER_SEQUENCE  = 3
sequence_length       = TARGET_FPS * SECONDS_PER_SEQUENCE  # 90 frames
videos_per_person     = 25

SAVE_ANNOTATED_VIDEO  = True

# =====================================
# حساب التقدم الإجمالي
# =====================================
def print_overall_progress():
    total = len(actions) * videos_per_person
    done  = sum(get_next_sequence_index(a, sequence_length) for a in actions)
    print(f"\n📊 Overall Progress: {done}/{total} sequences completed")
    print(f"   Remaining: {total - done} sequences\n")


# =====================================
# Webcam
# =====================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Camera not opened! Check your webcam index or permissions.")

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print_overall_progress()

# FIX: استخدام flag للخروج بدلاً من raise SystemExit داخل with
should_quit = False

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as hands:

    for action in actions:
        if should_quit:
            break

        idx      = action - BASE_CLASS_ID
        label    = signs[idx]
        eng_name = display_names[label]

        print(f"\nClass {action} → Sign: {label} ({eng_name})")

        start_seq = get_next_sequence_index(action, sequence_length)

        if start_seq >= videos_per_person:
            print(f"  ✅ Completed — {videos_per_person} sequences already recorded")
            continue

        for sequence in range(start_seq, videos_per_person):
            if should_quit:
                break

            print(f"\nRecording video {sequence}")
            print("ENTER = start | SPACE = pause/resume | Q = quit")

            # --------- Video folder + file ---------
            video_folder = os.path.join(VIDEO_PATH, str(action))
            os.makedirs(video_folder, exist_ok=True)
            video_path = os.path.join(video_folder, f"{sequence}.mp4")

            # --------- Keypoints folder ---------
            kp_path = os.path.join(DATA_PATH, str(action), str(sequence))
            os.makedirs(kp_path, exist_ok=True)

            fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, fourcc, TARGET_FPS, (frame_width, frame_height)
            )
            if not video_writer.isOpened():
                video_writer.release()
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("❌ VideoWriter not opened! Check codec/path permissions.")

            # -------- WAIT FOR ENTER --------
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                enhanced = enhance_image(frame)
                image, results = mediapipe_detection(enhanced, hands)
                draw_styled_landmarks(image, results)

                hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

                cv2.putText(image, f'{eng_name} | Video {sequence}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, 'Press ENTER to start | Q to quit',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Hands detected: {hand_count}',
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                cv2.imshow("OpenCV Feed", image)

                key = cv2.waitKey(1) & 0xFF
                if key == 13:   # ENTER
                    break
                if key == ord('q'):
                    # FIX: تنظيف موارد الـ sequence الحالي قبل الخروج
                    video_writer.release()
                    # حذف الـ sequence الجزئي الذي لم يبدأ بعد
                    if os.path.exists(kp_path) and not os.listdir(kp_path):
                        os.rmdir(kp_path)
                    should_quit = True
                    break

            if should_quit:
                video_writer.release()
                break

            # -------- RECORDING --------
            frame_num = 0
            paused    = False

            while frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    continue

                enhanced = enhance_image(frame)
                image, results = mediapipe_detection(enhanced, hands)
                draw_styled_landmarks(image, results)

                hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

                show_frame = image.copy()
                cv2.putText(show_frame,
                            f'{eng_name} | Video {sequence} | Frame {frame_num+1}/{sequence_length}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(show_frame, 'SPACE pause/resume | Q quit',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(show_frame, f'Hands detected: {hand_count}',
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

                if paused:
                    cv2.putText(show_frame, 'PAUSED',
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                cv2.imshow("OpenCV Feed", show_frame)

                k = cv2.waitKey(int(1000 / TARGET_FPS)) & 0xFF

                if k == ord('q'):
                    # FIX: حذف الـ sequence الجزئي عند الخروج أثناء التسجيل
                    video_writer.release()
                    print(f"  ⚠️  Exited while recording sequence {sequence} — it will be deleted")
                    for npy_file in os.listdir(kp_path):
                        os.remove(os.path.join(kp_path, npy_file))
                    os.rmdir(kp_path)
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    should_quit = True
                    break

                if k == 32:   # SPACE
                    paused = not paused

                if paused:
                    continue

                # Save video frame
                if SAVE_ANNOTATED_VIDEO:
                    video_writer.write(image)
                else:
                    video_writer.write(enhanced)

                # Save keypoints — shape (126,): left(63) + right(63) world coords
                kp = extract_keypoints(results)
                np.save(os.path.join(kp_path, str(frame_num)), kp)

                frame_num += 1

            # إغلاق الـ video writer بعد كل sequence
            video_writer.release()

            if not should_quit:
                print(f"  ✅ Sequence {sequence} saved successfully ({sequence_length} frames)")

# =====================================
# FIX: تنظيف موحد في نهاية البرنامج
# =====================================
cap.release()
cv2.destroyAllWindows()

if should_quit:
    print("\n⛔ Exited the program.")
else:
    print("\n🎉 Completed recording all sequences!")

print_overall_progress()