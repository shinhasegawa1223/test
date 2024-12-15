import mediapipe as mp
import cv2
import numpy as np

def get_landmarks(results, height, width):
    """
    Mediapipeの結果から特定のランドマーク座標を取得
    """
    keypoints = [11, 13, 15, 23, 25, 27]  # 左肩、左肘、左手首、左腰、左膝、左足首
    landmarks = {
        idx: (int(results.pose_landmarks.landmark[idx].x * width),
              int(results.pose_landmarks.landmark[idx].y * height))
        for idx in keypoints
    }
    return landmarks

def calc_distance(p1, p2, p3):
    """
    点p1から直線(p2, p3)への垂直距離を計算
    """
    u = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    return abs(np.cross(u, v) / np.linalg.norm(u))

def calc_slope(p1, p2):
    """
    2点間の傾きを計算
    """
    return np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)[0]

def is_low_pose(thresholds, slope, distances):
    """
    ローポーズの判定
    """
    slope_thresh, spine_dist_thresh, arm_dist_thresh = thresholds
    dist_hip, dist_knee, dist_elbow = distances
    return slope <= slope_thresh and \
           dist_hip < spine_dist_thresh and \
           dist_knee < spine_dist_thresh and \
           dist_elbow > arm_dist_thresh

def draw_landmarks(image, landmarks, settings):
    """
    ランドマークとそれらを結ぶ線を描画
    """
    radius, thickness, clr_kp, clr_line = settings
    points = [landmarks[idx] for idx in landmarks]

    for point in points:
        cv2.circle(image, point, radius, clr_kp, thickness)

    pairs = [(11, 13), (13, 15), (11, 23), (23, 25), (25, 27)]
    for p1, p2 in pairs:
        cv2.line(image, landmarks[p1], landmarks[p2], clr_line, thickness)

    return image

if __name__ == "__main__":
    # ======== 設定値 ========
    THRESHOLDS = (0, 30, 40)  # (傾斜, 背骨の距離, 腕の距離)
    DRAW_SETTINGS = (5, 2, (0, 0, 255), (255, 255, 255))  # (円の半径, 線の太さ, キーポイント色, ライン色)

    # Mediapipe Pose初期化
    mp_pose = mp.solutions.pose

    # 動画ファイルを読み込み
    cap = cv2.VideoCapture('training.mp4')

    # ======== Pose処理開始 ========
    with mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=False) as pose:
        low_pose_count = 0  # ローポーズのカウント
        low_pose_flag = False  # 現在ローポーズ中かどうか

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Empty frame")
                break

            # ======== 前処理 ========
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # 画像リサイズ
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGRをRGBに変換
            height, width = frame.shape[:2]  # 画像の高さと幅を取得

            # ======== ポーズ推定 ========
            results = pose.process(rgb_frame)

            if results.pose_landmarks:  # ランドマークが検出された場合のみ処理
                landmarks = get_landmarks(results, height, width)  # ランドマーク座標を取得
                print(landmarks)

                # ======== 必要な計算 ========
                distances = (
                    calc_distance(landmarks[11], landmarks[27], landmarks[23]),  # 肩-足首-腰の距離
                    calc_distance(landmarks[11], landmarks[27], landmarks[25]),  # 肩-足首-膝の距離
                    calc_distance(landmarks[11], landmarks[15], landmarks[13])   # 肩-手首-肘の距離
                )
                slope = calc_slope(landmarks[11], landmarks[27])  # 肩と足首の傾斜を計算

                # ======== ローポーズ判定 ========
                prev_flag = low_pose_flag
                low_pose_flag = is_low_pose(THRESHOLDS, slope, distances)

                if not prev_flag and low_pose_flag:  # 前フレームが非ローポーズで、現在がローポーズの場合カウント
                    low_pose_count += 1

                # ======== 描画 ========
                frame = draw_landmarks(frame, landmarks, DRAW_SETTINGS)  # ランドマークを描画
                cv2.putText(frame, str(low_pose_count), (20, 100),  # カウントを表示
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            # ======== 表示 ========
            cv2.imshow('AI Personal Trainer', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
                break

    # ======== 結果出力とリソース解放 ========
    print(f'合計 {low_pose_count} 回')
    cap.release()
    cv2.destroyAllWindows()
