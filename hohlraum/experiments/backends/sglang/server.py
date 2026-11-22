"""Experiment server entry; uses the unmodified SGLang launcher and scheduler."""
from functools import wraps
import os


def install_native_attachment(cache_type, attach):
    """Attach on a stock HiCache build after its backend is initialized."""
    if getattr(cache_type, '_hohlraum_native_attachment', False):
        return
    original = cache_type.init_hicache

    @wraps(original)
    def init_hicache(cache, *args, **kwargs):
        result = original(cache, *args, **kwargs)
        filename = os.environ.get('HOHLRAUM_CONFIG')
        # The bundled runtime already attaches. A None host means a
        # successful budget-only attachment.
        if filename and not hasattr(cache, 'hohlraum_host'):
            attach(cache, filename)
        return result

    cache_type.init_hicache = init_hicache
    cache_type._hohlraum_native_attachment = True


if os.environ.get('HOHLRAUM_CONFIG'):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from hohlraum.core.staging_storage import attach
    install_native_attachment(UnifiedRadixCache, attach)

# Executed in spawned workers as well as the parent launcher.
if os.environ.get('HOHLRAUM_NATIVE_FEEDBACK'):
    from hohlraum.experiments.backends.sglang.feedback import install
    install(os.environ['HOHLRAUM_NATIVE_FEEDBACK'])

if __name__ == '__main__':
    import sys
    from sglang.launch_server import run_server, prepare_server_args, load_plugins, kill_process_tree
    if not os.environ.get('HOHLRAUM_NATIVE_FEEDBACK'):
        raise SystemExit('Set HOHLRAUM_NATIVE_FEEDBACK to a new local experiment directory')
    load_plugins()
    try:
        if os.environ.get('HOHLRAUM_NATIVE_SERVER_ARGS'):
            import json
            from sglang.srt.server_args import ServerArgs
            args = ServerArgs(**json.loads(open(os.environ['HOHLRAUM_NATIVE_SERVER_ARGS']).read()))
        else:
            args = prepare_server_args(sys.argv[1:])
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
