"""Experiment server entry; uses the unmodified SGLang launcher and scheduler."""
import os

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
