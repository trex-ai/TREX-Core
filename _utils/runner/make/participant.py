import json

def make(configs, participant_id):
    script_path = '_clients/participants/sio_client.py'

    if 'server' not in configs:
        return None, None

    if 'participant' not in configs:
        return None, None

    participant_id = str(participant_id)
    if participant_id not in configs['participant']:
        return None, None

    participant_configs = configs['participant'][participant_id]

    if 'type' not in participant_configs:
        return None, None

    host = configs['server']['host'] if 'host' in configs['server'] else None
    port = str(configs['server']['port']) if 'port' in configs['server'] else None

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)

    args.append('--id=' + participant_id)
    args.append('--market_id=' + configs['market']['id'])
    args.append('--db_path=' + configs['study']['profiles_db_location'])
    args.append('--trader=' + json.dumps(participant_configs['trader']))

    if 'storage' in participant_configs:
        args.append('--storage=' + json.dumps(participant_configs['storage'])))

    if 'generation' in participant_configs and 'scale' in participant_configs['generation']:
        args.append('--generation_scale=' + str(participant_configs['generation']['scale']))

    if 'load' in participant_configs and 'scale' in participant_configs['load']:
        args.append('--load_scale=' + str(participant_configs['load']['scale']))

    return (script_path, args)
