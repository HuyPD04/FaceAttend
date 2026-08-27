from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from ..core.config import Settings


class InferenceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    detection_score: float
    quality_score: float
    face_pixels: int
    crop: np.ndarray
    sharpness: float
    brightness: float
    yaw_degrees: float
    roll_degrees: float


class FaceEngine:
    _strides = (8, 16, 32)
    _arcface_template = np.asarray(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, settings: Settings):
        self._settings = settings
        self._detector: ort.InferenceSession | None = None
        self._recognizer: ort.InferenceSession | None = None
        self._detector_input_name: str | None = None
        self._recognizer_input_name: str | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def warmup(self) -> None:
        self._ensure_loaded()

    def extract(self, image_bytes: bytes) -> list[DetectedFace]:
        self._ensure_loaded()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image is not a valid JPEG or PNG")
        if max(image.shape[:2]) > 1280:
            scale = 1280 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            detections = self._detect(image)
            return [self._to_detected_face(image, bbox, keypoints, score) for bbox, keypoints, score in detections]
        except InferenceUnavailableError:
            raise
        except Exception as exc:
            raise InferenceUnavailableError(f"Face inference failed: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            if not self._settings.detector_model_path.is_file():
                self._load_error = f"Detector model not found: {self._settings.detector_model_path}"
                raise InferenceUnavailableError(self._load_error)
            if not self._settings.recognizer_model_path.is_file():
                self._load_error = f"Recognizer model not found: {self._settings.recognizer_model_path}"
                raise InferenceUnavailableError(self._load_error)
            try:
                detector = ort.InferenceSession(
                    str(self._settings.detector_model_path),
                    providers=self._settings.provider_list,
                )
                recognizer = ort.InferenceSession(
                    str(self._settings.recognizer_model_path),
                    providers=self._settings.provider_list,
                )
                self._validate_detector(detector)
                self._validate_recognizer(recognizer)
                self._detector = detector
                self._recognizer = recognizer
                self._detector_input_name = detector.get_inputs()[0].name
                self._recognizer_input_name = recognizer.get_inputs()[0].name
                self._load_error = None
            except InferenceUnavailableError:
                raise
            except Exception as exc:
                self._load_error = str(exc)
                raise InferenceUnavailableError(
                    "Unable to load the configured SCRFD or ArcFace model"
                ) from exc

    def _detect(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        detector = self._detector
        input_name = self._detector_input_name
        if detector is None or input_name is None:
            raise InferenceUnavailableError("SCRFD detector is not loaded")
        input_size = self._settings.detector_input_size
        height, width = image.shape[:2]
        scale = min(input_size / width, input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        padded[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 128.0,
            size=(input_size, input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = detector.run(None, {input_name: blob})
        if len(outputs) != 9:
            raise InferenceUnavailableError(
                f"Unsupported SCRFD output count: {len(outputs)}; expected 9 outputs"
            )
        boxes: list[np.ndarray] = []
        keypoints: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for index, stride in enumerate(self._strides):
            score_values = np.asarray(outputs[index], dtype=np.float32).reshape(-1)
            bbox_values = np.asarray(outputs[index + 3], dtype=np.float32).reshape(-1, 4) * stride
            keypoint_values = (
                np.asarray(outputs[index + 6], dtype=np.float32).reshape(-1, 5, 2) * stride
            )
            centers = self._anchor_centers(input_size, stride, len(score_values))
            if len(centers) != len(score_values):
                raise InferenceUnavailableError("SCRFD output shape does not match its feature map")
            positive = np.where(score_values >= self._settings.detector_threshold)[0]
            if not len(positive):
                continue
            selected_centers = centers[positive]
            selected_boxes = self._distance_to_bbox(selected_centers, bbox_values[positive])
            selected_keypoints = selected_centers[:, None, :] + keypoint_values[positive]
            boxes.append(selected_boxes)
            keypoints.append(selected_keypoints)
            scores.append(score_values[positive])
        if not boxes:
            return []
        all_boxes = np.concatenate(boxes, axis=0) / scale
        all_keypoints = np.concatenate(keypoints, axis=0) / scale
        all_scores = np.concatenate(scores, axis=0)
        keep = self._nms(all_boxes, all_scores, self._settings.detector_nms_threshold)
        return [
            (all_boxes[index], all_keypoints[index], float(all_scores[index])) for index in keep
        ]

    def _to_detected_face(
        self, image: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, score: float
    ) -> DetectedFace:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Face crop is empty")
        face_pixels = min(x2 - x1, y2 - y1)
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        brightness = float(grayscale.mean())
        sharpness_score = min(sharpness / 100.0, 1.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        size_score = min(face_pixels / self._settings.min_face_pixels, 1.0)
        quality_score = float(min(score, sharpness_score, brightness_score, size_score))
        embedding = self._embed(image, keypoints)
        yaw_degrees, roll_degrees = self._estimate_pose(keypoints)
        return DetectedFace(
            embedding=embedding,
            detection_score=score,
            quality_score=quality_score,
            face_pixels=face_pixels,
            crop=crop,
            sharpness=sharpness,
            brightness=brightness,
            yaw_degrees=yaw_degrees,
            roll_degrees=roll_degrees,
        )

    def _embed(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        recognizer = self._recognizer
        input_name = self._recognizer_input_name
        if recognizer is None or input_name is None:
            raise InferenceUnavailableError("ArcFace recognizer is not loaded")
        transform, _ = cv2.estimateAffinePartial2D(
            keypoints.astype(np.float32), self._arcface_template, method=cv2.LMEDS
        )
        if transform is None:
            raise InferenceUnavailableError("Unable to align face landmarks for ArcFace")
        size = self._settings.recognizer_input_size
        aligned = cv2.warpAffine(image, transform, (size, size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1 / 127.5,
            size=(size, size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        embedding = np.asarray(recognizer.run(None, {input_name: blob})[0], dtype=np.float32).reshape(-1)
        if embedding.shape != (self._settings.embedding_dimension,):
            raise InferenceUnavailableError(
                f"ArcFace returned {embedding.shape[0]} dimensions, expected {self._settings.embedding_dimension}"
            )
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            raise InferenceUnavailableError("ArcFace returned a zero embedding")
        return np.ascontiguousarray(embedding / norm, dtype=np.float32)

    def _validate_detector(self, detector: ort.InferenceSession) -> None:
        inputs = detector.get_inputs()
        if len(inputs) != 1 or len(detector.get_outputs()) != 9:
            raise InferenceUnavailableError("Configured detector is not a compatible SCRFD 5-landmark model")

    def _validate_recognizer(self, recognizer: ort.InferenceSession) -> None:
        inputs = recognizer.get_inputs()
        if len(inputs) != 1 or not recognizer.get_outputs():
            raise InferenceUnavailableError("Configured recognizer is not a compatible ArcFace model")

    @staticmethod
    def _anchor_centers(input_size: int, stride: int, count: int) -> np.ndarray:
        feature_size = input_size // stride
        anchor_count = count // (feature_size * feature_size)
        if anchor_count < 1 or anchor_count * feature_size * feature_size != count:
            return np.empty((0, 2), dtype=np.float32)
        grid_y, grid_x = np.mgrid[:feature_size, :feature_size]
        centers = np.stack((grid_x, grid_y), axis=-1).astype(np.float32).reshape(-1, 2) * stride
        return np.repeat(centers, anchor_count, axis=0)

    @staticmethod
    def _distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                points[:, 0] - distances[:, 0],
                points[:, 1] - distances[:, 1],
                points[:, 0] + distances[:, 2],
                points[:, 1] + distances[:, 3],
            )
        )

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0, boxes[:, 3] - boxes[:, 1]
        )
        while len(order):
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if not len(remaining):
                break
            x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            union = areas[current] + areas[remaining] - intersection
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[overlap <= threshold]
        return keep

    @staticmethod
    def _estimate_pose(keypoints: np.ndarray) -> tuple[float, float]:
        left_eye, right_eye, nose = keypoints[:3].astype(np.float32)
        eye_vector = right_eye - left_eye
        eye_distance = max(float(np.linalg.norm(eye_vector)), 1.0)
        midpoint = (left_eye + right_eye) / 2
        yaw = float(np.clip((nose[0] - midpoint[0]) / eye_distance * 55.0, -45.0, 45.0))
        roll = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
        return yaw, roll
