# TrackerEncoder Inference Server

A FastAPI server that loads a single TrackerEncoder checkpoint and serves single-window predictions over HTTP.

## Starting the server

```bash
# From a wandb run directory (same as --base-folder in inference_video.py)
python server/server.py --wandb /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX

# With a specific checkpoint number (default: latest)
python server/server.py --wandb /path/to/wandb/run-... --checkpoint-number 10000

# From explicit config and checkpoint paths
python server/server.py --config /path/to/config.toml --checkpoint /path/to/checkpoint_00010000.pth

# Override device or port
python server/server.py --wandb /path/to/run --device cuda:1 --port 8080
```

The server prints `n_frames`, `image_size`, and `device` on startup.

## Endpoints

### `GET /info`

Returns model metadata. Useful for clients to discover `n_frames` before sending images.

**Response** `200 application/json`

```json
{
  "n_frames": 64,
  "image_size": 224,
  "device": "cuda:0",
  "config_path": "/path/to/config.toml",
  "checkpoint_path": "/path/to/checkpoint_00010000.pth",
  "mode_3d": "encoder"
}
```

---

### `POST /predict`

Runs one forward pass and returns all model outputs.

**Request** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | form field (JSON string) | Cameras, query coords, and optional query times (see below) |
| `images` | repeated file field | One image file per (camera, frame); filename must be `<cam_name>__<frame_idx>.<ext>` |

**`metadata` JSON schema**

```json
{
  "cameras": [
    {
      "name": "0",
      "type": "pinhole",
      "mat":    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist":   [k1, k2, p1, p2, k3],
      "ext":    [[r00,r01,r02,tx], [r10,r11,r12,ty], [r20,r21,r22,tz], [0,0,0,1]],
      "offset": [ox, oy]
    }
  ],
  "coords":      [[x, y, z], ...],
  "query_times": [t0, t1, ...]
}
```

- `cameras` — list of cameras in the order the model should consume them. Images are matched to cameras by `name`.
- `coords` — shape `(N, 3)`, 3D query points in world space (float).
- `query_times` — shape `(N,)`, integer frame index each query point originates from (optional; defaults to zeros).
- `ext` — 4×4 world-to-camera extrinsic matrix. The server computes `ext_inv` and `center` internally.
- `offset` — pixel offset applied when the image is a crop of a larger frame; use `[0, 0]` for full-frame images.
- Image dimensions (`size`) are read from the uploaded pixel data — the client does not need to include them.

**Image files**

- Filename format: `<cam_name>__<frame_idx>.<ext>` (e.g. `0__000064.png`).
- Every camera listed in `cameras` must have exactly `n_frames` files (where `n_frames` comes from `/info`).
- Any format readable by OpenCV is accepted (PNG, JPEG, …).
- Images are decoded as BGR and converted to RGB internally; send them in their natural on-disk format.

**Response** `200 application/octet-stream` — a NumPy `.npz` archive (filename `predictions.npz`).

Decode with:
```python
import io, numpy as np
result = np.load(io.BytesIO(response.content))
```

Output keys (shapes use `B=1`, `T=n_frames`, `N=num_query_points`, `C=num_cameras`):

| Key | Shape | Description |
|-----|-------|-------------|
| `coords_pred` | `(B, T, N, 3)` | Final 3D predictions (same as `3d_pred_direct`) |
| `3d_pred_direct` | `(B, T, N, 3)` | 3D predictions via direct regression |
| `3d_pred_rays` | `(B, T, N, 3)` | 3D predictions via ray-based aggregation |
| `3d_pred_triangulate` | `(B, T, N, 3)` | 3D predictions via triangulation (absent for single-camera input) |
| `3d_pred_cams_direct` | `(C, B, T, N, 3)` | Per-camera direct 3D predictions |
| `3d_pred_cams_rays` | `(C, B, T, N, 3)` | Per-camera ray-based 3D predictions |
| `2d_pred` | `(C, B, T, N, 2)` | Per-camera 2D reprojections |
| `vis_pred` | `(B, T, N, 1)` | Visibility logits (aggregated across cameras) |
| `conf_pred` | `(B, T, N, 1)` | Confidence logits (aggregated across cameras) |
| `vis_pred_2d` | `(C, B, T, N)` | Per-camera visibility logits |
| `conf_pred_2d` | `(C, B, T, N)` | Per-camera confidence logits |
| `conf_3d` | `(C, B, T, N)` | Per-camera 3D confidence logits |
| `depth_pred` | `(C, B, T, N)` | Per-camera depth predictions |

**Error responses** `400 text/plain` — short message describing which validation failed (missing camera images, wrong frame count, malformed filename, JSON parse error, etc.).

---

## Example client

`server/example_client.py` is a runnable end-to-end smoke test. It reads a trial directory (same layout as `inference_video.py`), sends one window of frames to the server, and prints shapes and NaN counts for each output key.

```bash
python server/example_client.py \
    --trial-path /path/to/session/trial/ \
    --server-url http://localhost:8000 \
    --start-frame 0
```
