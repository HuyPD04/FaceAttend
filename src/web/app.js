const state = {
  stream: null,
  cameraReady: false,
  scanning: false,
  scanTimer: null,
  scanInFlight: false,
  enrollment: null,
  enrollmentInFlight: false,
  enrollmentFeedback: null,
  clientId: localStorage.getItem("faceattend-client-id") || crypto.randomUUID(),
};

localStorage.setItem("faceattend-client-id", state.clientId);

const elements = {
  video: document.querySelector("#video"),
  placeholder: document.querySelector("#camera-placeholder"),
  startCamera: document.querySelector("#start-camera"),
  cameraStatus: document.querySelector("#camera-status"),
  health: document.querySelector("#health"),
  metrics: document.querySelector("#metrics"),
  scanToggle: document.querySelector("#scan-toggle"),
  result: document.querySelector("#recognition-result"),
  cameraId: document.querySelector("#camera-id"),
  direction: document.querySelector("#attendance-direction"),
  employeeForm: document.querySelector("#employee-form"),
  employees: document.querySelector("#employees"),
  events: document.querySelector("#attendance-events"),
  employeeTemplate: document.querySelector("#employee-template"),
  enrollmentWorkspace: document.querySelector("#enrollment-workspace"),
  enrollmentEmployee: document.querySelector("#enrollment-employee"),
  enrollmentProgress: document.querySelector("#enrollment-progress"),
  enrollmentResult: document.querySelector("#enrollment-result"),
  captureEnrollment: document.querySelector("#capture-enrollment"),
  completeEnrollment: document.querySelector("#complete-enrollment"),
  reviews: document.querySelector("#reviews"),
  reviewTemplate: document.querySelector("#review-template"),
};

function setResult(target, type, message, details = "") {
  target.className = `result ${type}`;
  target.textContent = details ? `${message} · ${details}` : message;
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.toggle("active", content.id === `${name}-tab`);
  });
}

function detailFromError(body) {
  if (typeof body.detail === "string") return body.detail;
  if (body.detail) return JSON.stringify(body.detail);
  return "Yêu cầu không thành công";
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(detailFromError(body));
  return body;
}

async function refreshHealth() {
  try {
    const health = await request("/api/health");
    elements.health.className = `health ${health.status === "ok" ? "ok" : "error"}`;
    const model = health.model_loaded ? "model sẵn sàng" : "model tải khi dùng";
    elements.health.textContent = `${health.mongo ? "MongoDB OK" : "MongoDB lỗi"} · ${health.faiss_vectors} vector · ${model}`;
  } catch (error) {
    elements.health.className = "health error";
    elements.health.textContent = error.message;
  }
}

async function refreshMetrics() {
  try {
    const metrics = await request("/api/metrics");
    const recognition = metrics.operations.recognition;
    if (!recognition) {
      elements.metrics.textContent = "Chưa có frame";
      return;
    }
    elements.metrics.textContent = `${recognition.requests} frame · p50 ${recognition.p50_latency_ms} ms · p95 ${recognition.p95_latency_ms} ms`;
  } catch (error) {
    elements.metrics.textContent = "Metrics chưa khả dụng";
  }
}

async function startCamera() {
  if (state.stream) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    elements.video.srcObject = state.stream;
    await elements.video.play();
    state.cameraReady = true;
    elements.placeholder.classList.add("hidden");
    elements.startCamera.textContent = "Camera đang bật";
    elements.startCamera.disabled = true;
    elements.scanToggle.disabled = false;
    elements.cameraStatus.textContent = "Đã kết nối webcam. Chỉ để một khuôn mặt trong khung hình.";
    renderEnrollment();
    await refreshEmployees();
  } catch (error) {
    elements.cameraStatus.textContent = `Không mở được camera: ${error.message}`;
  }
}

function captureFrame() {
  if (!state.cameraReady || !elements.video.videoWidth) {
    throw new Error("Hãy bật camera trước");
  }
  const canvas = document.createElement("canvas");
  const maxWidth = 960;
  const scale = Math.min(1, maxWidth / elements.video.videoWidth);
  canvas.width = Math.round(elements.video.videoWidth * scale);
  canvas.height = Math.round(elements.video.videoHeight * scale);
  canvas.getContext("2d").drawImage(elements.video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Không thể lấy frame từ camera"));
    }, "image/jpeg", 0.88);
  });
}

function recognitionDetails(result) {
  const details = [];
  if (result.full_name) details.push(`${result.full_name} (${result.employee_code})`);
  if (result.match_score != null) details.push(`match ${result.match_score.toFixed(3)}`);
  if (result.quality) details.push(`quality ${result.quality.score.toFixed(2)}`);
  if (result.confirmation_hits != null) details.push(`${result.confirmation_hits} frame`);
  return details.join(" · ");
}

async function scanOnce() {
  if (state.scanInFlight || !state.scanning) return;
  state.scanInFlight = true;
  try {
    const image = await captureFrame();
    const form = new FormData();
    form.append("image", image, "webcam.jpg");
    form.append("client_id", state.clientId);
    form.append("camera_id", elements.cameraId.value.trim() || "laptop-webcam");
    form.append("direction", elements.direction.value);
    const result = await request("/api/recognition/frame", { method: "POST", body: form });
    setResult(elements.result, result.status, result.message, recognitionDetails(result));
    if (result.event) refreshEvents();
    if (result.status === "unknown") refreshReviews();
  } catch (error) {
    setResult(elements.result, "error", error.message);
  } finally {
    state.scanInFlight = false;
    refreshMetrics();
  }
}

function toggleScanning() {
  state.scanning = !state.scanning;
  if (state.scanning) {
    elements.scanToggle.textContent = "Dừng quét";
    setResult(elements.result, "pending", "Đang quét. Hãy nhìn rõ vào camera.");
    scanOnce();
    state.scanTimer = window.setInterval(scanOnce, 450);
  } else {
    elements.scanToggle.textContent = "Bắt đầu quét";
    window.clearInterval(state.scanTimer);
    state.scanTimer = null;
    setResult(elements.result, "idle", "Đã dừng quét.");
  }
}

async function submitEmployee(event) {
  event.preventDefault();
  const payload = {
    employee_code: document.querySelector("#employee-code").value.trim(),
    full_name: document.querySelector("#employee-name").value.trim(),
    site_id: document.querySelector("#employee-site").value.trim(),
  };
  try {
    const employee = await request("/api/employees", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    elements.employeeForm.reset();
    document.querySelector("#employee-site").value = "default";
    await refreshEmployees();
    await startEnrollment(employee);
  } catch (error) {
    window.alert(error.message);
  }
}

function poseHint(session) {
  const hints = {
    front: "Nhìn thẳng camera để lấy mẫu chính diện.",
    left: "Quay nhẹ mặt sang trái rồi lấy mẫu tiếp theo.",
    right: "Quay nhẹ mặt sang phải rồi lấy mẫu tiếp theo.",
  };
  return hints[session.next_pose_hint] || hints.front;
}

function renderEnrollment() {
  const current = state.enrollment;
  if (!current) {
    elements.enrollmentWorkspace.classList.add("hidden");
    return;
  }
  const { employee, session } = current;
  elements.enrollmentWorkspace.classList.remove("hidden");
  elements.enrollmentEmployee.textContent = `${employee.full_name} (${employee.employee_code})`;
  elements.enrollmentProgress.textContent = `${session.sample_count}/${session.target_samples} mẫu`;
  const poses = session.pose_counts;
  const details = `Front ${poses.front || 0} · Left ${poses.left || 0} · Right ${poses.right || 0}`;
  const completed = session.status === "completed";
  const feedback = state.enrollmentFeedback || {
    type: completed ? "confirmed" : "idle",
    message: completed ? "Enrollment đã hoàn thành." : poseHint(session),
    details,
  };
  setResult(
    elements.enrollmentResult,
    completed ? "confirmed" : feedback.type,
    completed ? "Enrollment đã hoàn thành." : feedback.message,
    completed ? details : feedback.details,
  );
  elements.captureEnrollment.disabled = !state.cameraReady || completed || state.enrollmentInFlight;
  elements.completeEnrollment.disabled = completed || session.sample_count < session.min_samples || state.enrollmentInFlight;
}

async function startEnrollment(employee) {
  if (!state.cameraReady) {
    showTab("enrollment");
    window.alert("Hãy bật camera trước khi bắt đầu enrollment.");
    return;
  }
  try {
    const session = await request(`/api/employees/${employee.id}/enrollment-sessions`, { method: "POST" });
    state.enrollment = { employee, session };
    state.enrollmentFeedback = null;
    showTab("enrollment");
    renderEnrollment();
  } catch (error) {
    window.alert(error.message);
  }
}

async function captureEnrollment() {
  if (!state.enrollment || state.enrollmentInFlight) return;
  state.enrollmentInFlight = true;
  state.enrollmentFeedback = {
    type: "pending",
    message: "Đang kiểm tra chất lượng mẫu...",
    details: "",
  };
  renderEnrollment();
  try {
    const image = await captureFrame();
    const form = new FormData();
    form.append("image", image, "enrollment.jpg");
    const result = await request(
      `/api/enrollment-sessions/${state.enrollment.session.id}/frames`,
      { method: "POST", body: form },
    );
    state.enrollment.session = result.session;
    const detail = `quality ${result.quality.score.toFixed(2)} · ${result.quality.pose_bucket}`;
    state.enrollmentFeedback = {
      type: "confirmed",
      message: "Đã lưu mẫu khuôn mặt.",
      details: detail,
    };
    await refreshEmployees();
    await refreshHealth();
  } catch (error) {
    state.enrollmentFeedback = { type: "error", message: error.message, details: "" };
  } finally {
    state.enrollmentInFlight = false;
    renderEnrollment();
    refreshMetrics();
  }
}

async function completeEnrollment() {
  if (!state.enrollment || state.enrollmentInFlight) return;
  state.enrollmentInFlight = true;
  state.enrollmentFeedback = {
    type: "pending",
    message: "Đang hoàn tất enrollment...",
    details: "",
  };
  renderEnrollment();
  try {
    const session = await request(
      `/api/enrollment-sessions/${state.enrollment.session.id}/complete`,
      { method: "POST" },
    );
    state.enrollment.session = session;
    state.enrollmentFeedback = null;
    renderEnrollment();
  } catch (error) {
    state.enrollmentFeedback = { type: "error", message: error.message, details: "" };
  } finally {
    state.enrollmentInFlight = false;
    renderEnrollment();
  }
}

async function refreshEmployees() {
  try {
    const employees = await request("/api/employees");
    elements.employees.replaceChildren();
    if (!employees.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "Chưa có nhân viên.";
      elements.employees.append(empty);
      return;
    }
    employees.forEach((employee) => {
      const fragment = elements.employeeTemplate.content.cloneNode(true);
      fragment.querySelector(".employee-name").textContent = employee.full_name;
      fragment.querySelector(".employee-meta").textContent = `${employee.employee_code} · ${employee.site_id} · ${employee.status}`;
      fragment.querySelector(".template-count").textContent = `${employee.template_count} mẫu`;
      const button = fragment.querySelector(".enroll-button");
      button.disabled = employee.status !== "active" || !state.cameraReady;
      button.addEventListener("click", () => startEnrollment(employee));
      elements.employees.append(fragment);
    });
  } catch (error) {
    elements.employees.textContent = error.message;
  }
}

function formatDate(value) {
  return new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "medium" }).format(new Date(value));
}

async function refreshEvents() {
  try {
    const events = await request("/api/attendance");
    elements.events.replaceChildren();
    events.forEach((event) => {
      const row = document.createElement("tr");
      [
        formatDate(event.occurred_at),
        `${event.full_name} (${event.employee_code})`,
        event.event_type,
        event.camera_id,
        event.match_score.toFixed(3),
      ].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      elements.events.append(row);
    });
  } catch (error) {
    elements.events.textContent = error.message;
  }
}

async function dismissReview(reviewId) {
  try {
    await request(`/api/reviews/${reviewId}/dismiss`, { method: "POST" });
    await refreshReviews();
  } catch (error) {
    window.alert(error.message);
  }
}

async function refreshReviews() {
  try {
    const reviews = await request("/api/reviews");
    elements.reviews.replaceChildren();
    if (!reviews.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "Chưa có lượt cần duyệt.";
      elements.reviews.append(empty);
      return;
    }
    reviews.forEach((review) => {
      const fragment = elements.reviewTemplate.content.cloneNode(true);
      const image = fragment.querySelector(".review-evidence");
      if (review.evidence_available) {
        image.src = `/api/reviews/${review.id}/evidence`;
        image.classList.remove("hidden");
      }
      fragment.querySelector(".review-reason").textContent = review.reason;
      const score = review.match_score == null ? "không có match" : `match ${review.match_score.toFixed(3)}`;
      fragment.querySelector(".review-meta").textContent = `${formatDate(review.created_at)} · ${review.camera_id} · quality ${review.quality_score.toFixed(2)} · ${score}`;
      const dismiss = fragment.querySelector(".dismiss-review");
      dismiss.disabled = review.status !== "pending";
      dismiss.textContent = review.status === "pending" ? "Bỏ qua" : "Đã bỏ qua";
      dismiss.addEventListener("click", () => dismissReview(review.id));
      elements.reviews.append(fragment);
    });
  } catch (error) {
    elements.reviews.textContent = error.message;
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

elements.startCamera.addEventListener("click", startCamera);
elements.scanToggle.addEventListener("click", toggleScanning);
elements.employeeForm.addEventListener("submit", submitEmployee);
elements.captureEnrollment.addEventListener("click", captureEnrollment);
elements.completeEnrollment.addEventListener("click", completeEnrollment);
document.querySelector("#refresh-events").addEventListener("click", refreshEvents);
document.querySelector("#refresh-employees").addEventListener("click", refreshEmployees);
document.querySelector("#refresh-reviews").addEventListener("click", refreshReviews);

refreshHealth();
refreshMetrics();
refreshEmployees();
refreshEvents();
refreshReviews();
