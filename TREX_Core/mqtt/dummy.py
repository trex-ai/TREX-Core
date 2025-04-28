class DummyMQTT:
    """
    Drop‑in for gmqtt.Client during tests.
    Records every call to publish() so you can assert on it later.
    """
    def __init__(self):
        self.published = []          # (topic, payload, qos, retain, kwargs)

    def publish(self, topic, payload=None, qos=0, retain=False, **kw):
        # comment the next line if you don’t want console noise
        print(f"[dummy] publish to {topic!r} payload={payload!r}")
        self.published.append((topic, payload, qos, retain, kw))
        return None                  # gmqtt returns a MessageInfo, not needed

    # stubs for other gmqtt methods you never call in this test:
    def subscribe(self, *a, **kw):  pass
    def connect(self, *a, **kw):    pass
    def disconnect(self, *a, **kw): pass