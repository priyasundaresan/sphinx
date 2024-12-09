import json
import math
import asyncio
import websockets
import ast


# Function to adjust rotation values
def adjust_rotation(value):
    if math.isclose(value, -math.pi):  # Check if value is approximately -π
        return 0  # Adjust value to 0
    else:
        return value


async def websocket_handler(websocket, path):
    try:
        async for message in websocket:
            data = ast.literal_eval(message)
            orientation = data["orientation"]
            x_ori = data["orientation"]["x"]
            y_ori = data["orientation"]["y"]
            z_ori = data["orientation"]["z"]
            data["orientation"]["x"] = adjust_rotation(x_ori)
            data["orientation"]["y"] = adjust_rotation(y_ori)
            data["orientation"]["z"] = adjust_rotation(z_ori)
            data["gripper_open"] = data["url"] == "http://localhost:8080/franka.obj"
            del data["url"]
            print(data)
            # Your message processing logic here
            # For example, you can simply echo the message back to the client:
            await websocket.send(message)
    except websockets.exceptions.ConnectionClosedError:
        print("WebSocket connection closed")


def run_websocket_server(port=8765):
    start_server = websockets.serve(websocket_handler, "localhost", port)
    print(f"Starting WebSocket server on port {port}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    run_websocket_server()
