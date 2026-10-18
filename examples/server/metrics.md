#### `GET /metrics`

Returns server and generation statistics as JSON.

This endpoint is **disabled by default**. The server must be started with the
`--metrics` flag to enable it. Without the flag, the endpoint returns `501 Not
Implemented` with a plain-text message explaining how to enable it, regardless
of the `Accept` header.

With the flag enabled, the endpoint requires `Accept: application/json`. Returns
`406` otherwise.

Typical status codes:

- `200 OK`
- `406 Not Acceptable` (missing or wrong Accept header)
- `501 Not Implemented` (`--metrics` flag not passed)

Example response:

```json
{
  "server": {
    "uptime_seconds": 42.0,
    "timestamp": 1790580000
  },
  "processing": {
    "total_completed": 5,
    "total_failed": 0,
    "active": {
      "queued": 0,
      "generating": 1
    },
    "sync_completed": 2,
    "sync_failed": 0,
    "async_completed": 3,
    "async_failed": 0
  },
  "generation_speed": {
    "avg_seconds_per_image": 3.2,
    "images_per_second": 0.3125,
    "recent_seconds_per_image": {
      "avg_seconds_per_image": 1.95,
      "count": 1
    }
  },
  "model_memory": {
    "total_bytes": 683583532,
    "vram_bytes": 683583532,
    "ram_bytes": 0,
    "text_encoders_bytes": 166855680,
    "diffusion_model_bytes": 511829776,
    "vae_bytes": 4898076,
    "control_net_bytes": 0,
    "extensions_bytes": 0
  }
}
```

Field types:

| Field | Type | Notes |
| --- | --- | --- |
| `server.uptime_seconds` | `float` | Seconds since server start |
| `server.timestamp` | `integer` | Unix timestamp |
| `processing.total_completed` | `integer` | Total completed generations (sync + async combined) |
| `processing.total_failed` | `integer` | Total failed generations (sync + async combined) |
| `processing.active.queued` | `integer` | Async jobs waiting in queue |
| `processing.active.generating` | `integer` | Async jobs currently generating |
| `processing.sync_completed` | `integer` | Completed generations via synchronous endpoints |
| `processing.sync_failed` | `integer` | Failed generations via synchronous endpoints |
| `processing.async_completed` | `integer` | Completed generations via async endpoints |
| `processing.async_failed` | `integer` | Failed generations via async endpoints |
| `generation_speed.avg_seconds_per_image` | `float` | Overall average seconds per image (floored at 0.001s) |
| `generation_speed.images_per_second` | `float` | Overall images per second (derived from floored average) |
| `generation_speed.recent_seconds_per_image` | `object` | Rolling 10-job window — see below |
| `model_memory` | `object | null` | Model memory stats; `null` if no model loaded |

`recent_seconds_per_image`:

| Field | Type | Notes |
| --- | --- | --- |
| `avg_seconds_per_image` | `float` | Average over the last `count` jobs (floored at 0.001s) |
| `count` | `integer` | Number of async jobs in the rolling window (0–10) |

`model_memory` (experimental):

| Field | Type | Notes |
| --- | --- | --- |
| `total_bytes` | `integer` | Total model parameter memory (bytes) |
| `vram_bytes` | `integer` | Memory on GPU/device memory |
| `ram_bytes` | `integer` | Memory on system RAM |
| `text_encoders_bytes` | `integer` | Text encoder parameter memory |
| `diffusion_model_bytes` | `integer` | Diffusion model parameter memory |
| `vae_bytes` | `integer` | VAE parameter memory |
| `control_net_bytes` | `integer` | ControlNet parameter memory |
| `extensions_bytes` | `integer` | Extension/PhotoMaker parameter memory |

All memory sizes are in bytes.

**Experimental**: accuracy depends on backend and device. Not tested on all
model families. Use at your own risk.


