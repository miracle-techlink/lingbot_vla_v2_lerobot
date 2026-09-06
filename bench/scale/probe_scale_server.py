# Scaling-experiment server: official deploy stack, cwd redirected so that
# reset() picks up robot configs from $PROBE_CWD/configs/robot_configs/.
# Optional PROBE_DECOMP=1 wraps embed_prefix / prefix forward / predict_velocity
# with synchronized wall timers (intended for --use_compile false runs).
import os, sys, time

sys.path.insert(0, '/root/lingbot-vla-v2-official')
PROBE_CWD = os.environ.get('PROBE_CWD', '/root/d4090_official/scale/cwd')
os.chdir(PROBE_CWD)

import torch
import deploy.lingbot_vla_v2_policy as P

DECOMP = os.environ.get('PROBE_DECOMP', '0') == '1'

if DECOMP:
    _orig_load_vla = P.LingbotVLAv2Server.load_vla

    def _timed(name):
        def deco(fn):
            def wrapper(*a, **kw):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn(*a, **kw)
                torch.cuda.synchronize()
                print(f'[decomp] {name} {(time.perf_counter()-t0)*1000:.1f} ms', flush=True)
                return out
            return wrapper
        return deco

    def load_vla(self, path_to_pi_model):
        vla = _orig_load_vla(self, path_to_pi_model)
        m = vla.model
        m.embed_prefix = _timed('embed_prefix')(m.embed_prefix)
        m.predict_velocity = _timed('predict_velocity')(m.predict_velocity)
        _orig_fwd = m.qwenvl_with_expert.forward
        def fwd(*a, **kw):
            tag = 'prefix_fwd' if kw.get('fill_kv_cache', False) else 'denoise_fwd'
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _orig_fwd(*a, **kw)
            torch.cuda.synchronize()
            print(f'[decomp] {tag} {(time.perf_counter()-t0)*1000:.1f} ms', flush=True)
            return out
        m.qwenvl_with_expert.forward = fwd
        print('[decomp] wrappers installed', flush=True)
        return vla

    P.LingbotVLAv2Server.load_vla = load_vla

P.main()
