# Usage: scale_client.py <port> <robo_name> <ncam> <n_capture_calls> [tag]
import sys, time
sys.path.insert(0, '/root/lingbot-vla-v2-official')
import numpy as np
from deploy.websocket_client_policy import WebsocketClientPolicy

port = int(sys.argv[1]); robo = sys.argv[2]; ncam = int(sys.argv[3])
ncap = int(sys.argv[4]) if len(sys.argv) > 4 else 3
tag = sys.argv[5] if len(sys.argv) > 5 else robo

CAMS = ['observation.images.cam_high',
        'observation.images.cam_left_wrist',
        'observation.images.cam_right_wrist'][:ncam]

client = WebsocketClientPolicy(host='127.0.0.1', port=port)
t0 = time.time()
client.reset(robo)
print(f'reset({robo}) ok in {time.time()-t0:.1f}s', flush=True)

rng = np.random.default_rng(0)
def make_obs():
    obs = {'observation.state': rng.random(14).astype(np.float32),
           'task': 'pick up the red block'}
    for c in CAMS:
        obs[c] = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    return obs

t0 = time.perf_counter()
client.infer(make_obs())
print(f'COLD_CALL_S {time.perf_counter()-t0:.1f}', flush=True)
for _ in range(3):
    client.infer(make_obs())

walls = []
for i in range(ncap):
    t0 = time.perf_counter()
    out = client.infer(make_obs())
    dt = (time.perf_counter() - t0) * 1000
    walls.append(dt)
    st = out.get('server_timing', {})
    print(f'CAPTURE_CALL {i} wall={dt:.1f}ms server_infer={st.get("infer_ms", float("nan")):.1f}ms', flush=True)

print(f'RESULT {tag} ncam={ncam} wall_mean={np.mean(walls):.1f}', flush=True)
print('CLIENT_DONE', flush=True)
