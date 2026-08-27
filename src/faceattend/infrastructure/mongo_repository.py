from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from bson import ObjectId
from bson.binary import Binary
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from ..core.config import Settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NotFoundError(Exception):
    pass


class DuplicateEmployeeError(Exception):
    pass


class MongoRepository:
    def __init__(self, settings: Settings):
        self.client: MongoClient[dict[str, Any]] = MongoClient(
            settings.mongodb_uri,
            tz_aware=True,
            serverSelectionTimeoutMS=3_000,
        )
        self.db: Database[dict[str, Any]] = self.client[settings.mongodb_database]
        self.employees: Collection[dict[str, Any]] = self.db["employees"]
        self.templates: Collection[dict[str, Any]] = self.db["face_templates"]
        self.attendance: Collection[dict[str, Any]] = self.db["attendance_events"]
        self.enrollment_sessions: Collection[dict[str, Any]] = self.db["enrollment_sessions"]
        self.reviews: Collection[dict[str, Any]] = self.db["recognition_reviews"]
        self.counters: Collection[dict[str, Any]] = self.db["counters"]

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def ensure_indexes(self) -> None:
        self.employees.create_index([("employee_code", ASCENDING)], unique=True)
        self.employees.create_index([("status", ASCENDING), ("site_id", ASCENDING)])
        self.templates.create_index([("faiss_id", ASCENDING)], unique=True)
        self.templates.create_index([("employee_id", ASCENDING), ("active", ASCENDING)])
        self.attendance.create_index([("idempotency_key", ASCENDING)], unique=True)
        self.attendance.create_index([("occurred_at", DESCENDING)])
        self.attendance.create_index([("employee_id", ASCENDING), ("occurred_at", DESCENDING)])
        self.enrollment_sessions.create_index([("employee_id", ASCENDING), ("status", ASCENDING)])
        self.enrollment_sessions.create_index([("created_at", DESCENDING)])
        self.reviews.create_index([("dedup_key", ASCENDING)], unique=True)
        self.reviews.create_index([("status", ASCENDING), ("created_at", DESCENDING)])

    @staticmethod
    def _object_id(value: str) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise NotFoundError("Invalid object id")
        return ObjectId(value)

    @staticmethod
    def _employee_read(document: dict[str, Any], template_count: int) -> dict[str, Any]:
        return {
            "id": str(document["_id"]),
            "employee_code": document["employee_code"],
            "full_name": document["full_name"],
            "site_id": document["site_id"],
            "status": document["status"],
            "template_count": template_count,
            "created_at": document["created_at"],
        }

    def create_employee(self, employee_code: str, full_name: str, site_id: str) -> dict[str, Any]:
        document = {
            "employee_code": employee_code.upper(),
            "full_name": full_name.strip(),
            "site_id": site_id.strip(),
            "status": "active",
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        try:
            result = self.employees.insert_one(document)
        except DuplicateKeyError as exc:
            raise DuplicateEmployeeError("Employee code already exists") from exc
        document["_id"] = result.inserted_id
        return self._employee_read(document, 0)

    def list_employees(self) -> list[dict[str, Any]]:
        documents = list(self.employees.find().sort("created_at", DESCENDING))
        if not documents:
            return []
        counts = {
            row["_id"]: row["count"]
            for row in self.templates.aggregate(
                [
                    {"$match": {"active": True}},
                    {"$group": {"_id": "$employee_id", "count": {"$sum": 1}}},
                ]
            )
        }
        return [self._employee_read(item, counts.get(item["_id"], 0)) for item in documents]

    def get_employee(self, employee_id: str) -> dict[str, Any]:
        document = self.employees.find_one({"_id": self._object_id(employee_id)})
        if not document:
            raise NotFoundError("Employee was not found")
        template_count = self.templates.count_documents(
            {"employee_id": document["_id"], "active": True}
        )
        return self._employee_read(document, template_count)

    def reserve_faiss_id(self) -> int:
        counter = self.counters.find_one_and_update(
            {"_id": "faiss_template_id"},
            {"$inc": {"sequence": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(counter["sequence"])

    def create_template(
        self,
        *,
        employee_id: str,
        faiss_id: int,
        embedding_encrypted: bytes,
        quality_score: float,
        model_version: str,
        embedding_model_id: str,
        detector_model_id: str,
        preprocess_version: str,
        embedding_dimension: int,
    ) -> dict[str, Any]:
        employee_object_id = self._object_id(employee_id)
        employee = self.employees.find_one({"_id": employee_object_id, "status": "active"})
        if not employee:
            raise NotFoundError("Active employee was not found")
        document = {
            "employee_id": employee_object_id,
            "faiss_id": faiss_id,
            "embedding_encrypted": Binary(embedding_encrypted),
            "embedding_dimension": embedding_dimension,
            "quality_score": quality_score,
            "model_version": model_version,
            "embedding_model_id": embedding_model_id,
            "detector_model_id": detector_model_id,
            "preprocess_version": preprocess_version,
            "active": True,
            "created_at": utcnow(),
        }
        result = self.templates.insert_one(document)
        return {
            "template_id": str(result.inserted_id),
            "employee_id": employee_id,
            "quality_score": quality_score,
            "model_version": model_version,
            "faiss_id": faiss_id,
        }

    def delete_template(self, faiss_id: int) -> None:
        self.templates.delete_one({"faiss_id": faiss_id})

    @staticmethod
    def enrollment_session_read(document: dict[str, Any]) -> dict[str, Any]:
        pose_counts = document.get("pose_counts", {})
        return {
            "id": str(document["_id"]),
            "employee_id": str(document["employee_id"]),
            "status": document["status"],
            "target_samples": document["target_samples"],
            "min_samples": document["min_samples"],
            "sample_count": document["sample_count"],
            "pose_counts": pose_counts,
            "next_pose_hint": MongoRepository._next_pose_hint(pose_counts),
            "created_at": document["created_at"],
            "completed_at": document.get("completed_at"),
        }

    @staticmethod
    def _next_pose_hint(pose_counts: dict[str, int]) -> str:
        if pose_counts.get("front", 0) == 0:
            return "front"
        if pose_counts.get("left", 0) == 0:
            return "left"
        if pose_counts.get("right", 0) == 0:
            return "right"
        return "front"

    def start_enrollment_session(
        self,
        employee_id: str,
        target_samples: int,
        min_samples: int,
    ) -> dict[str, Any]:
        employee_object_id = self._object_id(employee_id)
        employee = self.employees.find_one({"_id": employee_object_id, "status": "active"})
        if not employee:
            raise NotFoundError("Active employee was not found")
        active = self.enrollment_sessions.find_one(
            {"employee_id": employee_object_id, "status": "active"}, sort=[("created_at", DESCENDING)]
        )
        if active:
            return self.enrollment_session_read(active)
        document = {
            "employee_id": employee_object_id,
            "status": "active",
            "target_samples": target_samples,
            "min_samples": min_samples,
            "sample_count": 0,
            "pose_counts": {"front": 0, "left": 0, "right": 0},
            "samples": [],
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        result = self.enrollment_sessions.insert_one(document)
        document["_id"] = result.inserted_id
        return self.enrollment_session_read(document)

    def get_enrollment_session(self, session_id: str) -> dict[str, Any]:
        document = self.enrollment_sessions.find_one({"_id": self._object_id(session_id)})
        if not document:
            raise NotFoundError("Enrollment session was not found")
        return document

    def record_enrollment_sample(
        self,
        *,
        session_id: str,
        template_id: str,
        quality_score: float,
        pose_bucket: str,
        yaw_degrees: float,
        roll_degrees: float,
    ) -> dict[str, Any]:
        session = self.get_enrollment_session(session_id)
        if session["status"] != "active":
            raise ValueError("Enrollment session is already completed")
        if session["sample_count"] >= session["target_samples"]:
            raise ValueError("Enrollment session already has its target sample count")
        sample_count = session["sample_count"] + 1
        completed = sample_count >= session["target_samples"]
        update = {
            "$push": {
                "samples": {
                    "template_id": template_id,
                    "quality_score": quality_score,
                    "pose_bucket": pose_bucket,
                    "yaw_degrees": yaw_degrees,
                    "roll_degrees": roll_degrees,
                    "created_at": utcnow(),
                }
            },
            "$inc": {"sample_count": 1, f"pose_counts.{pose_bucket}": 1},
            "$set": {"updated_at": utcnow()},
        }
        if completed:
            update["$set"].update({"status": "completed", "completed_at": utcnow()})
        updated = self.enrollment_sessions.find_one_and_update(
            {"_id": session["_id"], "status": "active", "sample_count": session["sample_count"]},
            update,
            return_document=ReturnDocument.AFTER,
        )
        if not updated:
            raise ValueError("Enrollment session changed; retry the sample")
        return self.enrollment_session_read(updated)

    def complete_enrollment_session(self, session_id: str) -> dict[str, Any]:
        session = self.get_enrollment_session(session_id)
        if session["status"] == "completed":
            return self.enrollment_session_read(session)
        if session["sample_count"] < session["min_samples"]:
            raise ValueError("Capture the minimum number of samples before completing enrollment")
        updated = self.enrollment_sessions.find_one_and_update(
            {"_id": session["_id"], "status": "active"},
            {"$set": {"status": "completed", "completed_at": utcnow(), "updated_at": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if not updated:
            raise ValueError("Enrollment session changed; retry completion")
        return self.enrollment_session_read(updated)

    def active_template_rows(self) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [
            {"$match": {"active": True}},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "employee_id",
                    "foreignField": "_id",
                    "as": "employee",
                }
            },
            {"$unwind": "$employee"},
            {"$match": {"employee.status": "active"}},
            {
                "$project": {
                    "faiss_id": 1,
                    "embedding_encrypted": 1,
                    "employee_id": 1,
                    "employee_code": "$employee.employee_code",
                    "full_name": "$employee.full_name",
                }
            },
        ]
        return list(self.templates.aggregate(pipeline))

    def deactivate_employee(self, employee_id: str) -> list[int]:
        employee_object_id = self._object_id(employee_id)
        template_ids = [
            row["faiss_id"]
            for row in self.templates.find(
                {"employee_id": employee_object_id, "active": True}, {"faiss_id": 1}
            )
        ]
        result = self.employees.update_one(
            {"_id": employee_object_id, "status": "active"},
            {"$set": {"status": "inactive", "updated_at": utcnow()}},
        )
        if not result.matched_count:
            raise NotFoundError("Active employee was not found")
        self.templates.update_many(
            {"employee_id": employee_object_id, "active": True},
            {"$set": {"active": False, "revoked_at": utcnow()}},
        )
        return template_ids

    def find_template_by_faiss_id(self, faiss_id: int) -> dict[str, Any] | None:
        return self.templates.find_one({"faiss_id": faiss_id, "active": True})

    def get_employee_by_object_id(self, employee_id: ObjectId) -> dict[str, Any] | None:
        return self.employees.find_one({"_id": employee_id, "status": "active"})

    def find_last_event(self, employee_id: str, camera_id: str) -> dict[str, Any] | None:
        return self.attendance.find_one(
            {"employee_id": self._object_id(employee_id), "camera_id": camera_id},
            sort=[("occurred_at", DESCENDING)],
        )

    def create_attendance_event(self, document: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            result = self.attendance.insert_one(document)
            document["_id"] = result.inserted_id
            return document, True
        except DuplicateKeyError:
            existing = self.attendance.find_one({"idempotency_key": document["idempotency_key"]})
            if not existing:
                raise
            return existing, False

    @staticmethod
    def attendance_read(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(document["_id"]),
            "employee_id": str(document["employee_id"]),
            "employee_code": document["employee_code"],
            "full_name": document["full_name"],
            "camera_id": document["camera_id"],
            "event_type": document["event_type"],
            "occurred_at": document["occurred_at"],
            "match_score": document["match_score"],
            "quality_score": document["quality_score"],
            "model_version": document["model_version"],
        }

    def list_attendance(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.attendance.find().sort("occurred_at", DESCENDING).limit(limit)
        return [self.attendance_read(row) for row in rows]

    def create_review(self, document: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.reviews.insert_one(document)
            document["_id"] = result.inserted_id
            return document
        except DuplicateKeyError:
            existing = self.reviews.find_one({"dedup_key": document["dedup_key"]})
            if not existing:
                raise
            return existing

    @staticmethod
    def review_read(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(document["_id"]),
            "status": document["status"],
            "reason": document["reason"],
            "camera_id": document["camera_id"],
            "quality_score": document["quality_score"],
            "match_score": document.get("match_score"),
            "candidate_employee_code": document.get("candidate_employee_code"),
            "evidence_available": bool(document.get("evidence_encrypted")),
            "created_at": document["created_at"],
        }

    def list_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.reviews.find().sort("created_at", DESCENDING).limit(limit)
        return [self.review_read(row) for row in rows]

    def dismiss_review(self, review_id: str) -> dict[str, Any]:
        updated = self.reviews.find_one_and_update(
            {"_id": self._object_id(review_id), "status": "pending"},
            {"$set": {"status": "dismissed", "dismissed_at": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if not updated:
            raise NotFoundError("Pending review was not found")
        return self.review_read(updated)

    def get_review_evidence(self, review_id: str) -> bytes | None:
        document = self.reviews.find_one({"_id": self._object_id(review_id)}, {"evidence_encrypted": 1})
        if not document:
            raise NotFoundError("Review was not found")
        evidence = document.get("evidence_encrypted")
        return bytes(evidence) if evidence else None

    def close(self) -> None:
        self.client.close()
