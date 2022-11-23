from pathlib import Path
def cli(configs):
<<<<<<< HEAD
    # TODO: Nov 30, 2021; this fixes the path problem

    path_to_trex = str(Path(configs['study']['sim_root']))
    path_to_trex = configs['study']['sim_root']
    script_path = path_to_trex + '_server/sio_server.py'
    # script_path = '_server/sio_server.py'

=======
    path = __file__.split('_utils')
    script_path = path[0] + '_server/sio_server.py'
>>>>>>> Package
    if 'server' not in configs:
        print('server not in configs')
        return None, None

    host = configs['server']['host']
    port = str(configs['server']['port'])

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)
    return (script_path, args)
