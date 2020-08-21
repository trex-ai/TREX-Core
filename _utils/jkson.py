import rapidjson as json

def dumps(obj, **kwargs):
    return json.dumps(obj)

def loads(s, **kwargs):
    return json.loads(s)