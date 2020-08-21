source_info = {
    # consolidate source specific information in here
    # also used to classify the type by simply checking if 'key' exists in dict
    "solar": "non_dispatch",
    "wind": "non_dispatch",
    "bess": "dispatch"
}

async def classify(source):
    source = source.lower()
    if source in source_info:
        return source_info[source]
    return None