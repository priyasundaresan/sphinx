import pyrealsense2 as rs

ctx = rs.context()  # type: ignore
if len(ctx.devices) > 0:
    for d in ctx.devices:
        print(
            "Found device: ",
            d.get_info(rs.camera_info.name),  # type: ignore
            " ",
            d.get_info(rs.camera_info.serial_number),  # type: ignore
        )
else:
    print("No Intel Device connected")
