import json
import os
# from pathlib import

# this is the makefile for a synchronous policy server,
# where the collection of experience and the learning on that experience happen in sequence and not in parallel
# as such, only one client is needed

# Fixme: the name that is used in the makefile here does not read from the config file?
def cli(configs):
    path = __file__.split('runner')
    script_path = os.path.join(path[0], 'appo', 'async_actor.py')
    if not os.path.exists(script_path):
        path = os.getcwd()
        script_path = os.path.join(path, 'clients', 'appo', 'async_actor.py')
    # print('policy server', script_path)
    if not os.path.exists(script_path):
        return None, None

    if 'server' not in configs:
        return None, None

    host = configs['server']['host']
    port = str(configs['server']['port'])

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)
    args.append('--config=' + json.dumps(configs))
    return (script_path, args)
