import json


def cli(configs):
    module_path = "TREX_Core.sim_controller.client"

    if "server" not in configs:
        return None, None

    host = configs["server"]["host"]
    port = str(configs["server"]["port"])

    args = []
    if host:
        args.append("--host=" + host)
    if port:
        args.append("--port=" + port)
    args.append("--config=" + json.dumps(configs))
    return (module_path, args)
