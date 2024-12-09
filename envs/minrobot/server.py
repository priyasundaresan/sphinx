import zerorpc
import torch
import time
import polymetis


class PolyMetisServer:
    """A minimal server that runs on the nuc.

    -- It always performs positional control.
    -- The client (running on workstation) is responsible for any advanced control.
    -- The client is responsible for safety, range checks.
    -- All external facing functions receive and return python native types for
        compatibility with 0rpc.
    -- All internal facing functions use torch.Tensor for compatibility with polymetis.
    """

    def __init__(self) -> None:
        self.robot_ip = "localhost"

    def hello(self):
        return "hello"

    def set_home_joints(self, home_joints: list[float]):
        self._home_joints = torch.tensor(home_joints, dtype=torch.float32)

    def init_robot(self):
        self._robot = polymetis.RobotInterface(ip_address=self.robot_ip, enforce_version=False)
        self._gripper = polymetis.GripperInterface(ip_address=self.robot_ip)
        if hasattr(self._gripper, "metadata") and hasattr(self._gripper.metadata, "max_width"):
            # Should grab this from robotiq2f
            self._max_gripper_width = self._gripper.metadata.max_width
        else:
            self._max_gripper_width = 0.08  # default, from FrankaHand Value

        # reset gripper & go home
        self.update_gripper(1, blocking=True)
        self._go_home(self._home_joints)
        ee_pos = self._robot.get_ee_pose()[0]
        print(f"init ee pos: {ee_pos}")

    def update(self, pos: list[float], quat: list[float], gripper_open: float) -> None:
        """
        Updates the robot controller with the action
        """
        assert len(pos) == 3 and len(quat) == 4

        if not self._robot.is_running_policy():
            print("restarting cartesian impedance controller")
            self._robot.start_cartesian_impedance()
            time.sleep(1)

        self._robot.update_desired_ee_pose(
            torch.tensor(pos, dtype=torch.float32), torch.tensor(quat, dtype=torch.float32)
        )

        # Update the gripper
        for _ in range(3):
            self.update_gripper(gripper_open, blocking=True)

    def update_gripper(self, gripper_open: float, blocking=False) -> None:
        # We always run the gripper in absolute position
        gripper_open = max(min(gripper_open, 1), 0)
        width = self._max_gripper_width * gripper_open

        self._gripper.goto(
            width=width,
            speed=0.5,
            force=0.1,
            blocking=blocking,
        )

    def _go_home(self, home_joints: torch.Tensor):
        """home is in joint position"""
        self._robot.set_home_pose(home_joints)
        self._robot.go_home(blocking=True)

    def reset(self, randomize: bool) -> None:
        print("reset env")

        # open the gripper
        self.update_gripper(1, blocking=True)

        if self._robot.is_running_policy():
            self._robot.terminate_current_policy()

        home_joints = self._home_joints
        if randomize:
            # TODO: make it adjustable
            noise = 0.1 * (2 * torch.rand_like(self._home_joints) - 1)
            print("home noise:", noise)
            home_joints += noise

        self._go_home(home_joints)

        assert not self._robot.is_running_policy()
        self._robot.start_cartesian_impedance()
        time.sleep(1)

    def get_proprio(self) -> dict:
        """
        Returns the robot state dictionary.
        """
        ee_pos, ee_quat = self._robot.get_ee_pose()
        gripper_state = self._gripper.get_state()
        gripper_open = gripper_state.width / self._max_gripper_width

        state = {
            "eef_pos": ee_pos.tolist(),
            "eef_quat": ee_quat.tolist(),
            "gripper_open": gripper_open,
        }

        return state


def main():
    server = PolyMetisServer()
    s = zerorpc.Server(server)
    s.bind("tcp://0.0.0.0:4242")
    s.run()


if __name__ == "__main__":
    torch.set_printoptions(precision=4, linewidth=100, sci_mode=False)
    main()
