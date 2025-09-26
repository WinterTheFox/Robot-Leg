import pyrealsense2 as rs

ctx = rs.context()
if len(ctx.devices) == 0:
    print("❌ No se encontró una cámara RealSense.")
else:
    for dev in ctx.devices:
        sensors = dev.query_sensors()
        for sensor in sensors:
            print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
            for profile in sensor.get_stream_profiles():
                try:
                    vsp = profile.as_video_stream_profile()
                    fmt = vsp.format()
                    print(f"  - {rs.format(fmt).name} {vsp.width()}x{vsp.height()} @ {profile.fps()} FPS")
                except:
                    continue