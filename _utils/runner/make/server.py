from pathlib import Path
def cli(configs):
    # TODO: Nov 30, 2021; this fixes the path problem
    path_to_trex = str(Path('C:/source/Trex-Core/'))
    script_path = path_to_trex + '/_server/sio_client.py'
    # script_path = '_server/sio_server.py'

    if 'server' not in configs:
        return None, None

    host = configs['server']['host'] if 'host' in configs['server'] else None
    port = str(configs['server']['port']) if 'port' in configs['server'] else None

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)
    return (script_path, args)
