# PAL Lab Log

## 2025-01-25

### real world robot


Let me analyze the workspace structure and key interfaces for controlling the robot:

1. **Main Control Interfaces**:
The project has two main control modes:
- Waypoint Mode: Discrete position control
- Dense Mode: Continuous position control

2. **Key Entry Points**:

a) **Testing Basic Movement**:

```7:36:envs/minrobot/test_controller.py
def goto(target_x, target_y, target_z, delta, max_step):
    cfg = ControllerConfig(server_ip="tcp://172.16.0.1:4242")
    client = Controller(cfg)
    client.reset(False)

    proprio = client.get_proprio()
    x, y, z = proprio.eef_pos

    for _ in range(max_step):
        proprio = client.get_proprio()
        x, y, z = proprio.eef_pos
        print(f"{x:3f}, {y:.3f}, {z:.3f}")

        if abs(target_x - x) < 0.02 and abs(target_y - y) < 0.02 and abs(target_z - z) < 0.02:
            break

        delta_pos = np.array(
            [
                np.sign(target_x - x) * delta,
                np.sign(target_y - y) * delta,
                np.sign(target_z - z) * delta,
            ]
        )
        delta_euler = np.array([0, 0, 0])
        gripper_open = 0.5

        client.delta_control(np.array(delta_pos), np.array(delta_euler), gripper_open)
        time.sleep(0.1)

    return
```

This provides the simplest interface to test robot movement. You could adapt this for R2D2:
```python
def test_r2d2_movement():
    # Basic position control test
    target_pos = [0.307, 0, 0.586]  # Example target
    delta = 0.01  # Movement step size
    max_steps = 100
    
    # Your R2D2 controller implementation
    client = R2D2Controller()
    client.reset()
    
    # Move to target
    current_pos = client.get_position()
    while steps < max_steps:
        delta_pos = calculate_delta(current_pos, target_pos, delta)
        client.move(delta_pos)
```

3. **Core Robot Environment**:

```14:234:envs/franka_env.py
class FrankaEnv:
    def __init__(self, cfg: FrankaEnvConfig):
        self.cfg = cfg

        self.controller = Controller(cfg.controller)
        self.controller.reset(randomize=False)
        self.controller.reset(randomize=False)
        proprio = self.controller.get_proprio()
        self.home_pos = proprio.eef_pos
        self.home_euler = proprio.eef_euler

        if cfg.parallel_camera:
            self.camera = ParallelCameras(cfg.cameras)
        else:
            self.camera = SequentialCameras(cfg.cameras)


    def reset(self):
        self.move_to(self.home_pos, self.home_euler, gripper_open=1, control_freq=10)
        self.controller.reset(bool(self.cfg.randomize_init))

    def observe_camera(self):
        obs = {}
        show_images = []  # for rendering

        cam_frames = self.camera.get_frames()


        for name, frames in cam_frames.items():
            show_images.append(frames["image"])

            for k, v in frames.items():
                if self.cfg.channel_first:
                    v = v.transpose(2, 0, 1)
                obs[f"{name}_{k}"] = v

        if self.cfg.show_camera:
            image = np.hstack(show_images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        return obs
    def observe_proprio(self) -> Proprio:
        return self.controller.get_proprio()

    def observe(self):
        obs = self.observe_camera()

        proprio = self.controller.get_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["ee_quat"] = proprio.eef_quat
        obs["gripper_open"] = np.array([proprio.gripper_open])
        obs["proprio"] = proprio.eef_pos_euler

        return obs

    def apply_action(
        self, ee_pos: np.ndarray, ee_euler: np.ndarray, gripper_open: float, is_delta=True
    ):
        if is_delta:
            self.controller.delta_control(ee_pos, ee_euler, gripper_open)
        else:
            self.controller.position_control(ee_pos, ee_euler, gripper_open)
    def move_to(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        control_freq: float,
        recorder: Optional[DatasetRecorder] = None,
        max_delta=0.01,
        plot=False,
    ):
        proprio = self.controller.get_proprio()
        curr_gripper_open = float(proprio.gripper_open > 0.95)

        # functions for positional and rotational interpolation
        gen_waypoint, num_steps = get_waypoint(proprio.eef_pos, target_pos, max_delta=max_delta)
        gen_ori = get_ori(proprio.eef_euler, target_euler, num_steps)

        if plot:
            pos_plotter = MoveErrorPlot(target_pos)
            # rot_plotter = MoveErrorPlot(target_euler)

        for i in range(1, num_steps + 1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)

            with common_utils.FreqGuard(control_freq):
                if recorder is not None:
                    obs = self.observe()
                    delta_pos, delta_euler = self.controller.position_to_delta(
                        next_ee_pos, next_ee_euler
                    )
                    action = np.concatenate([delta_pos, delta_euler, [curr_gripper_open]])
                    action = action.astype(np.float32)
                    recorder.record(ActMode.Interpolate, obs, action)

                self.controller.position_control(next_ee_pos, next_ee_euler, curr_gripper_open)

                if plot:
                    delta_pos, delta_euler = self.controller.position_to_delta(
                        next_ee_pos, next_ee_euler
                    )
                    pp = self.observe_proprio()
                    # print("delta pos norm:", np.linalg.norm(target_pos - pp.eef_pos))
                    pos_plotter.add(pp.eef_pos, target_pos, delta_pos)
    ...
    def update_gripper(self, gripper_open, control_freq, recorder: Optional[DatasetRecorder]):
        prev_width = self.observe_proprio().gripper_open
        if np.abs(prev_width - gripper_open) == 0:
            return

        while True:
            with common_utils.FreqGuard(control_freq):
                if recorder is not None:
                    obs = self.observe()

                self.apply_action(np.zeros(3), np.zeros(3), gripper_open)

                if recorder is not None:
                    action = np.concatenate([np.zeros(3), np.zeros(3), [gripper_open]])
                    recorder.record(ActMode.Interpolate, obs, action)

            curr_width = self.observe_proprio().gripper_open
            if np.abs(curr_width - prev_width) < 0.002:
                # exit when there is no change
                return

            prev_width = curr_width
    def move_to_acc(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: float,
        control_freq: float,
        recorder: Optional[DatasetRecorder] = None,
        plot=False,
    ):
        wpr_cfg = WaypointReachConfig()
        wpr_cfg.pos_max_norm = 0.025
        wpr_cfg.pos_threshold = 0.015
        wpr_cfg.rot_max_norm = 0.1
        wpr_cfg.rot_threshold = 0.075

        waypoint_reach = WaypointReach(
            np.ones(6),  # no scaling
            target_pos,
            target_euler,
            wpr_cfg,
        )

        if plot:
            pos_plotter = MoveErrorPlot(target_pos)
            # rot_plotter = MoveErrorPlot(target_euler)

        for i in range(50):
            proprio = self.controller.get_proprio()
            delta_pos, delta_euler, reached = waypoint_reach.step(
                proprio.eef_pos, proprio.eef_euler
            )

            if reached:
                break

            with common_utils.FreqGuard(control_freq):
                # if recorder is not None:
                #     obs = self.observe()
                #     delta_pos, delta_euler = self.controller.position_to_delta(
                #         next_ee_pos, next_ee_euler
                #     )
                #     action = np.concatenate(
                #         [delta_pos, delta_euler, [self.curr_gripper_open]]
                #     ).astype(np.float32)
                #     recorder.record(ActMode.Interpolate, obs, action)

                self.controller.delta_control(delta_pos, delta_euler, self.curr_gripper_open)

                if plot:
                    # delta_pos, delta_euler = self.controller.position_to_delta(
                    #     next_ee_pos, next_ee_euler
                    # )
                    pos_plotter.add(self.observe_proprio().eef_pos, target_pos, delta_pos)
...
def move():
    import pyrallis

    cfg = pyrallis.parse(config_class=FrankaEnvConfig)  # type: ignore
    env = FrankaEnv(cfg)

    proprio = env.observe_proprio()
    env.move_to_acc(
        proprio.eef_pos + np.array([0.2, -0.1, -0.1]),
        proprio.eef_euler + np.array([0.2, -0.2, -0.2]),
        1,
        10,
        plot=True,
    )
    del env
```

This shows the main robot environment interface with key methods:
- `move_to()`: Position control
- `apply_action()`: Action execution
- `observe()`: Get robot state

4. **Action Space**:
The code expects actions in this format:
- Position: [x, y, z]
- Orientation: Euler angles [rx, ry, rz]
- Gripper: Single value [0-1]

5. **Configuration Requirements**:
You'll need to create:
- Robot configuration file similar to `envs/fr3.yaml`
- Camera calibration if using vision
- Workspace bounds definition

6. **Suggested Integration Path**:

a) First implement basic controller:
```python
class R2D2Controller:
    def get_proprio(self):
        # Return current robot state
        return Proprio(
            eef_pos=[x, y, z],
            eef_quat=[qx, qy, qz, qw],
            gripper_open=gripper_state
        )
    
    def delta_control(self, delta_pos, delta_euler, gripper_open):
        # Implement relative motion
        pass
```

b) Test with basic movement script
c) Implement full environment wrapper
d) Integrate with UI if needed

Would you like me to elaborate on any of these aspects or provide more specific implementation details for any part?


### perception

1. tools/vis