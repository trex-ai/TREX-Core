import json

def cli(configs):
    script_path = '_clients/markets/sio_client.py'

    if 'server' not in configs:
        return None, None

    if 'market' not in configs:
        return None, None

    host = configs['server']['host']
    port = str(configs['server']['port'])

    market_configs = configs['market']
    market_configs['timezone'] = configs['study']['timezone']

    # TODO: temporarily add method to manually define profile step size until auto detection works
    if 'time_step_size' in configs['study']:
        market_configs['time_step_size'] = configs['study']['time_step_size']

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)

    args.append('--configs=' + json.dumps(market_configs))
    return (script_path, args)
