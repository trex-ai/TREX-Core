import json
import os
# from pathlib import Path
def cli(configs):
    path = __file__.split('runner')
    script_path = os.path.join(path[0], 'pettingzoo', 'client.py')
    if not os.path.exists(script_path):
        path = os.getcwd()
        script_path = os.path.join(path, 'clients', 'pettingzoo', 'client.py')
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
