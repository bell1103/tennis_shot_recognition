"""
Capture shots from annotation as a succession of features into a csv files
Note that we dont save useless features like eyes and ears positions.
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from extract_human_pose import (
    HumanPoseExtractor,
)

columns = [
    "nose_y",
    "nose_x",
    "left_shoulder_y",
    "left_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_x",
    "left_elbow_y",
    "left_elbow_x",
    "right_elbow_y",
    "right_elbow_x",
    "left_wrist_y",
    "left_wrist_x",
    "right_wrist_y",
    "right_wrist_x",
    "left_hip_y",
    "left_hip_x",
    "right_hip_y",
    "right_hip_x",
    "left_knee_y",
    "left_knee_x",
    "right_knee_y",
    "right_knee_x",
    "left_ankle_y",
    "left_ankle_x",
    "right_ankle_y",
    "right_ankle_x",
]

# MoveNet keypoint order (17 total):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
# We keep: 0 (nose) + 5-16 (shoulders to ankles) = 13 keypoints
KEEP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def extract_features(features):
    """
    Extract the 13 relevant keypoints (y, x) from the 17-keypoint array.
    features shape: (17, 3) — [y, x, confidence]
    Returns shape: (1, 26)
    """
    kept = features[KEEP_INDICES]  # shape (13, 3)
    return kept[:, 0:2].reshape(1, 13 * 2)  # shape (1, 26)


def draw_shot(frame, shot):
    """Draw shot name on frame (user-friendly)"""
    cv2.putText(
        frame,
        shot,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )
    print(f"Capturing {shot}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate (associate human pose to a tennis shot)"
    )
    parser.add_argument("video")
    parser.add_argument("annotation")
    parser.add_argument("out")
    parser.add_argument(
        "--show",
        action="store_const",
        const=True,
        default=False,
        help="Show frame",
    )
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Show sub frame",
    )
    args = parser.parse_args()

    shots = pd.read_csv(args.annotation)
    CURRENT_ROW = 0

    NB_IMAGES = 30
    shots_features = []

    FRAME_ID = 1
    IDX_FOREHAND = 1
    IDX_BACKHAND = 1
    IDX_NEUTRAL = 1
    IDX_SERVE = 1

    cap = cv2.VideoCapture(args.video)

    assert cap.isOpened()

    ret, frame = cap.read()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if CURRENT_ROW >= len(shots):
            print("Done, no more shots in annotation!")
            break

        human_pose_extractor.extract(frame)

        # dont draw non-significant points/edges by setting probability to 0
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2 == FRAME_ID:
            shots_features = []

        if (
            shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
            <= FRAME_ID
            <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
        ):
            if np.mean(features[:, 2]) < 0.3:
                CURRENT_ROW += 1
                shots_features = []
                print("Cancel this shot")
                FRAME_ID += 1
                continue

            features_row = extract_features(features)  # always (1, 26)

            shot_class = shots.iloc[CURRENT_ROW]["Shot"]
            shots_features.append(features_row)
            draw_shot(frame, shot_class)

            if FRAME_ID - NB_IMAGES // 2 + 1 == shots.iloc[CURRENT_ROW]["FrameId"]:
                shots_df = pd.DataFrame(
                    np.concatenate(shots_features, axis=0),
                    columns=columns,
                )
                shots_df["shot"] = np.full(NB_IMAGES, shot_class)

                if shot_class == "forehand-volley":
                    outpath = Path(args.out).joinpath(
                        f"forehand_{IDX_FOREHAND:03d}.csv"
                    )
                    IDX_FOREHAND += 1
                elif shot_class == "backhand-volley":
                    outpath = Path(args.out).joinpath(
                        f"backhand_{IDX_BACKHAND:03d}.csv"
                    )
                    IDX_BACKHAND += 1
                elif shot_class == "serve":
                    outpath = Path(args.out).joinpath(f"serve_{IDX_SERVE:03d}.csv")
                    IDX_SERVE += 1
                else:
                    print(f"Unknown shot class '{shot_class}', skipping save.")
                    CURRENT_ROW += 1
                    shots_features = []
                    FRAME_ID += 1
                    continue

                shots_df.to_csv(outpath, index=False)
                assert len(shots_df) == NB_IMAGES
                print(f"saving {shot_class} to {outpath}")

                CURRENT_ROW += 1
                shots_features = []

        # Guard against CURRENT_ROW == 0 to avoid iloc[-1] access
        elif CURRENT_ROW > 0 and (
            shots.iloc[CURRENT_ROW]["FrameId"] - shots.iloc[CURRENT_ROW - 1]["FrameId"]
            > NB_IMAGES
        ):
            frame_id_between_shots = (
                shots.iloc[CURRENT_ROW - 1]["FrameId"]
                + shots.iloc[CURRENT_ROW]["FrameId"]
            ) // 2
            if (
                frame_id_between_shots - NB_IMAGES // 2
                < FRAME_ID
                <= frame_id_between_shots + NB_IMAGES // 2
            ):
                features_row = extract_features(features)  # always (1, 26)
                shots_features.append(features_row)
                draw_shot(frame, "neutral")

                if FRAME_ID == frame_id_between_shots + NB_IMAGES // 2:
                    shots_df = pd.DataFrame(
                        np.concatenate(shots_features, axis=0),
                        columns=columns,
                    )
                    shots_df["shot"] = np.full(NB_IMAGES, "neutral")
                    outpath = Path(args.out).joinpath(f"neutral_{IDX_NEUTRAL:03d}.csv")
                    print(f"saving neutral to {outpath}")
                    IDX_NEUTRAL += 1
                    shots_df.to_csv(outpath, index=False)
                    shots_features = []

        # Display results on original frame
        if args.show:
            human_pose_extractor.draw_results_frame(frame)
            cv2.imshow("Frame", frame)

        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

        FRAME_ID += 1

    cap.release()
    cv2.destroyAllWindows()