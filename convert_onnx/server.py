from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort

app = FastAPI()

# عدّلي المسار حسب اسم مودلك
MODEL_PATH = "model.onnx"

# حمّل المودل مرة واحدة
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


class PredictRequest(BaseModel):
    # landmarks: shape = [90][42] floats
    landmarks: list[list[float]] = Field(..., description="90x42 sequence")


class PredictResponse(BaseModel):
    index: int
    confidence: float
    scores_len: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lm = req.landmarks

    # تحقق من الأبعاد
    if len(lm) != 90:
        raise HTTPException(status_code=400, detail=f"Expected 90 frames, got {len(lm)}")

    for i, row in enumerate(lm):
        if len(row) != 42:
            raise HTTPException(status_code=400, detail=f"Frame {i}: Expected 42 features, got {len(row)}")

    # (1, 90, 42) float32
    x = np.array(lm, dtype=np.float32)[None, :, :]

    # inference
    out = sess.run([output_name], {input_name: x})[0]  # shape غالبًا (1, num_classes)
    scores = out[0].astype(np.float32)

    idx = int(np.argmax(scores))
    conf = float(scores[idx])

    return PredictResponse(index=idx, confidence=conf, scores_len=int(scores.shape[0]))
