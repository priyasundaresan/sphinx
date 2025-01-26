import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class GELLOInterface:
    """
    A driver class for GELLO haptic device.
    
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
        action_scale (float): Scale factor for actions
    """
    def __init__(
        self,
        pos_sensitivity=1.0,
        rot_sensitivity=4.0,
        action_scale=0.08,
    ):
        print("Initializing GELLO device")
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.action_scale = action_scale
        
        # Initialize device-specific connection here
        # self.device = ...
        
        # State variables
        self.gripper_is_closed = False
        self._control = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.lock_state = 0
        self.rotation = np.eye(3)  # Initial rotation matrix
        
        # Launch listener thread
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def _display_controls(self):
        """Print the control mapping for users."""
        print("\nGELLO Control Mapping:")
        # Add your control descriptions here
        
    def start_control(self):
        """Initialize controller state before receiving commands."""
        self._control = np.zeros(6)
        self.lock_state = 0
        self.rotation = np.eye(3)
        
    def get_controller_state(self):
        """
        Get current state of the GELLO device.
        
        Returns:
            dict: Contains position deltas, rotation, gripper state, etc.
        """
        dpos = self.control[:3] * self.pos_sensitivity
        rotation = self.control[3:] * self.rot_sensitivity
        
        return {
            "dpos": dpos,
            "rotation": self.rotation,
            "raw_drotation": rotation,
            "grasp": self.control_gripper,
            "lock": self.lock_state,
        }
        
    def run(self):
        """Main loop to read device data."""
        while True:
            # Read data from your device
            # Update self._control, gripper state, etc.
            # Example:
            # device_data = self.device.read()
            # self._control = process_device_data(device_data)
            time.sleep(0.01)  # Adjust rate as needed
            
    @property
    def control(self):
        """Get current 6-DoF control values."""
        return np.array(self._control)
        
    @property 
    def control_gripper(self):
        """Get current gripper state."""
        return self.gripper_is_closed
        
    def get_action(self):
        """
        Get scaled actions from device state.
        
        Returns:
            tuple: (scaled_control, gripper_state, lock_state)
        """
        if np.any(np.abs(self.control) > 0.0) or self.control_gripper is not None:
            return (
                self.action_scale * self.control,
                self.control_gripper,
                self.lock_state,
            )
        return None, self.control_gripper, self.lock_state

# You can create a similar class for VR controllers
class VRInterface:
    """
    A driver class for VR controllers.
    """
    def __init__(self):
        # Similar structure to GELLOInterface
        pass

if __name__ == "__main__":
    interface = GELLOInterface()
    interface.start_control()
    while True:
        data = interface.get_controller_state()
        print(f"Position delta: {data['dpos'].round(3)}")
        time.sleep(0.1)