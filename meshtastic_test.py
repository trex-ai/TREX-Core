import meshtastic
import meshtastic.serial_interface
from pubsub import pub
import msgpack

count = 0
last_count = 0
def onReceive(packet, interface): # called when a packet arrives
    global count
    global last_count
    # print(f"Received: {packet}")
    if 'decoded' in packet:
        decoded = packet['decoded']
        portnum = decoded['portnum']
        payload = decoded['payload']
        match portnum:
            # case 'PRIVATE_APP':
            case 'ZPS_APP':
                count += 1
                payload = msgpack.loads(payload)['count']
                # print(payload)
                print(payload)
                if (payload - last_count) > 1:
                    print("desync")
                last_count = payload

def onConnection(interface, topic=pub.AUTO_TOPIC): # called when we (re)connect to the radio
    # defaults to broadcast, specify a destination ID if you wish
    interface.sendText("hello mesh")

pub.subscribe(onReceive, "meshtastic.receive")
# pub.subscribe(onConnection, "meshtastic.connection.established")
# By default will try to find a meshtastic device, otherwise provide a device path like /dev/ttyUSB0
interface = meshtastic.serial_interface.SerialInterface('COM5')
# interface.sendData(msgpack.dumps({"data": "hello mesh 4"}))
# interface.close()
