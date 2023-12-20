import json
from pathlib import Path
def cli(configs):
<<<<<<< HEAD
    # TODO: Nov 30, 2021; this fixes the path problem
    path_to_trex = str(Path('C:/source/Trex-Core/'))
    script_path = path_to_trex + '/_clients/sim_controller/sio_client.py'
    # script_path = '_clients/sim_controller/sio_client.py'
=======
    path = __file__.split('_utils')

    script_path = path[0] + '_clients/sim_controller/sio_client.py'
>>>>>>> Package

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
