import threading
import time
from collections import namedtuple
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    Examples:

        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    Examples:

        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True
        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


try:
    import os

    os.environ["LD_LIBRARY_PATH"] = os.getcwd()  # or whatever path you want
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. \n"
        "Installation:"
        " https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html?highlight=spacemouse"
    ) from exc


AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.
    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte
    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.
    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling
    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.
    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte
    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouseInterface:
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.
    Args:
        vendor_id (int): HID device vendor id
        product_id (int): HID device product id
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling

    See https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html?highlight=spacemouse
    """

    def __init__(
        self,
        vendor_id=9583,
        product_id=50734,
        pos_sensitivity=1.0,
        rot_sensitivity=4.0,
        action_scale=0.08,
    ):
        print("Opening SpaceMouse device")
        # print(hid.enumerate())
        # print(vendor_id, product_id)
        self.device = hid.device()
        self.device.open(vendor_id, product_id)  # SpaceMouse

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.action_scale = action_scale

        self.gripper_is_closed = False

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False
        self.elapsed_time = 0

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.lock_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "toggle gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._control = np.zeros(6)

        self.single_click_and_hold = False
        self.t_last_click = time.time()

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self.lock_state = 0

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            hold=self.single_click_and_hold,
            lock=self.lock_state,
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        # t_last_click = -1

        while True:
            d = self.device.read(13)
            #print(d)

            if d is not None:
                # readings from 6-DoF sensor
                if d[0] == 1:
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                    self.roll = -1.0 * convert(d[11], d[12])
                    self.pitch = convert(d[7], d[8])
                    self.yaw = convert(d[9], d[10])

                    self._control = [
                        self.x,
                        self.y,
                        self.z,
                        self.yaw,
                        self.pitch,
                        self.roll,
                    ]

                # readings from the side buttons

                elif d[0] == 3:
                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        self.elapsed_time = t_click - self.t_last_click
                        self.t_last_click = t_click
                        if self.elapsed_time > 0.5:
                            self.single_click_and_hold = True
                            self.gripper_is_closed = not self.gripper_is_closed

                    # release button
                    if d[1] == 0:
                        self.single_click_and_hold = False

                    # right button is for reset
                    if d[1] == 2 + int(self.single_click_and_hold):
                        self.lock_state = 1
                    else:
                        self.lock_state = 0

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """

        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        return self.gripper_is_closed

    def get_action(self):
        if sum(abs(self.control)) > 0.0 or self.control_gripper is not None:
            return (
                self.action_scale * self.control,
                self.control_gripper,
                self.lock_state,
            )
        else:
            return None, self.control_gripper, self.lock_state


if __name__ == "__main__":
    interface = SpaceMouseInterface()
    interface.start_control()
    while True:
        data = interface.get_controller_state()
        positional_delta = data["dpos"]
        # print(data['raw_drotation'].round(3))
        # rotational_delta = R.from_matrix(data['rotation']).as_euler('xyz', degrees=False)
        # print(rotational_delta.round(3))
        # print(data)
        # print(data['dpos'].round(3))
