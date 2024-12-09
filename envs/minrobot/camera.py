# import threading
# from abc import abstractmethod, abstractproperty
# from typing import Dict, Optional, Union

import numpy as np
import cv2
import pyrealsense2 as rs

import multiprocessing as mp


"""
NOTE: All cameras are set to record at 640x480 and then resize to the desired height and width.
"""


class RealSenseCamera:
    def __init__(
        self,
        serial_number: str,
        width: int,
        height: int,
        use_depth: bool,
        exposure=-1,
        record_width=640,
        record_height=480,
    ):
        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.serial_number = serial_number
        self.exposure = exposure

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, record_width, record_height, rs.format.rgb8, 30)


        if self.use_depth:
            config.enable_stream(
                rs.stream.depth, record_width, record_height, rs.format.z16, 30
            )
            self.depth_filters = [
                rs.spatial_filter().process,
                rs.temporal_filter().process,
                # rs.decimation_filter().process,
                # rs.hole_filling_filter().process
            ]
        else:
            self.depth_filters = []

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # # Get color sensors
        # sensors = self.pipeline.get_active_profile().get_device().query_sensors()
        # for sensor in sensors:
        #     if sensor.is_depth_sensor():
        #         sensor.set_option(rs.option.enable_auto_exposure, False)
        #         sensor.set_option(rs.option.exposure, 3000)

        #for sensor in self.pipeline.get_active_profile().get_device().sensors:
        #    if sensor.is_color_sensor():
        #        sensor.set_option(rs.option.enable_auto_white_balance, 1.0)

        # warmup cameras
        # TODO: better warm up
        for _ in range(2):
            self.pipeline.wait_for_frames()

    def get_intrinsics(self):
        profile = self.pipeline.get_active_profile()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        cprofile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        cintrinsics = cprofile.get_intrinsics()
        return dict(
            matrix=np.array(
                [
                    [cintrinsics.fx, 0, cintrinsics.ppx],
                    [0, cintrinsics.fy, cintrinsics.ppy],
                    [0, 0, 1.0],
                ]
            ),
            width=cintrinsics.width,
            height=cintrinsics.height,
            depth_scale=depth_scale,
        )

    def get_frames(self) -> dict[str, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        image = np.asanyarray(aligned_frames.get_color_frame().get_data())
        # DON'T SKIP THE NEXT LINE we need resize, which implicitly copies
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frames = dict(image=image)

        if self.use_depth:
            depth = aligned_frames.get_depth_frame()
            for rs_filter in self.depth_filters:
                depth = rs_filter(depth)

            depth = np.asanyarray(depth.get_data())
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth
        return frames

    def close(self):
        self.pipeline.stop()


class SequentialCameras:
    def __init__(self, camera_args_list: list[dict]):
        self.name_to_camera_args = {}
        self.cameras: dict[str, RealSenseCamera] = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.name_to_camera_args[name] = camera_args
            self.cameras[name] = RealSenseCamera(**camera_args)

    def get_intrinsics(self, name):
        return self.cameras[name].get_intrinsics()

    def get_frames(self):
        frames = {}
        for name, camera in self.cameras.items():
            frames[name] = camera.get_frames()
        return frames

    def __del__(self):
        # FIXME: well
        for _, camera in self.cameras.items():
            camera.close()


class ParallelCameras:
    def __init__(self, camera_args_list: list[dict]):
        # self.camera_args_list = camera_args_list
        self.name_to_camera_args = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.name_to_camera_args[name] = camera_args

        self.camera_procs = {}
        self.put_queues = {}
        self.get_queues = {}
        self.intrinsics = {}

        for name, camera_args in self.name_to_camera_args.items():
            put_queue = mp.Queue(maxsize=1)
            get_queue = mp.Queue(maxsize=1)
            proc = mp.Process(target=self._camera_proc, args=(camera_args, put_queue, get_queue))
            proc.start()
            self.camera_procs[name] = proc

            self.intrinsics[name] = get_queue.get()
            print(f"cam {name} constructed")

            self.put_queues[name] = put_queue
            self.get_queues[name] = get_queue

    def _camera_proc(self, camera_args, receive_queue: mp.Queue, send_queue: mp.Queue):
        camera = RealSenseCamera(**camera_args)
        send_queue.put(camera.get_intrinsics())

        while True:
            msg = receive_queue.get()
            if msg == "terminate":
                break

            assert msg == "get"
            assert send_queue.empty()

            frames = camera.get_frames()
            send_queue.put(frames)

        camera.close()

    def get_intrinsics(self, name):
        return self.intrinsics[name]

    def get_frames(self):
        for _, put_queue in self.put_queues.items():
            assert put_queue.empty()
            put_queue.put("get")

        camera_frames = {}
        for name, get_queue in self.get_queues.items():
            camera_frames[name] = get_queue.get()

        return camera_frames

    def __del__(self):
        # FIXME: well
        for name, put_queue in self.put_queues.items():
            print(f"terminating {name}")
            put_queue.put("terminate")
            self.camera_procs[name].join()
