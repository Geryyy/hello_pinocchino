import mujoco
from mujoco.viewer import launch_passive
import os
import numpy as np
import time
import pinocchio as pin

# Path to your MuJoCo XML model file
xml_file = '../robot_model/robocrane.xml'

# Ensure the file exists
if not os.path.exists(xml_file):
    raise FileNotFoundError(f"XML file not found: {xml_file}")

# Load the MuJoCo model from the XML file
mj_model = mujoco.MjModel.from_xml_path(xml_file)
mj_data = mujoco.MjData(mj_model)


### CREATE THE PINOCCHINO ###
# Be cautious of textures in the XML file, as they may lead to problems.
pin_model = pin.buildModelFromMJCF(xml_file)
pin_data = pin_model.createData()
print("Full DoF count: ", pin_model.nq)

# print names of all the joints
for i in range(pin_model.njoints):
    print(pin_model.names[i])

# Reduced model: Lock the joints from the 10th joint onwards (i.e. gripper joints)
joints_to_lock = [joint_name for joint_name in pin_model.names[10:]]

joints_to_lock_ids = []
for jn in joints_to_lock:
    if pin_model.existJointName(jn):
        joints_to_lock_ids.append(pin_model.getJointId(jn))
    else:
        print("Warning: joint " + str(jn) + " does not belong to the model!")

initial_joint_config = np.zeros(pin_model.nq)

pin_model_red = pin.buildReducedModel(pin_model, joints_to_lock_ids, initial_joint_config)
nq = pin_model_red.nq
print("reduced model nq: ", nq)

pin_data_red = pin_model_red.createData()


### END PINOCCHINO ###


framerate = 100
frame_time = 1.0 / framerate
print("frame_time: ", frame_time)
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    start = time.time()
    last_frame_time = time.time()
    while viewer.is_running():
        # Record the start time of each iteration for timekeeping
        step_start = time.time()

        mujoco.mj_step(mj_model, mj_data)

        ### PINOCCHINO ###
        
        q = np.array(mj_data.qpos[0:nq])
        v = np.array(mj_data.qvel[0:nq])
        a = np.array(mj_data.qacc[0:nq])

        # compute all kinematic and dynamic terms
        pin.forwardKinematics(pin_model_red, pin_data_red, q, v, a)
        pin.computeAllTerms(pin_model_red, pin_data_red, q, v)
        # you get them in pin_data_red

        # or you can compute them one by one

        # forward kinematics
        pin.framesForwardKinematics(pin_model_red, pin_data_red, q)
        # joint coordinate frames oMi
        H_joint_se3 = pin_data_red.oMi[-1] # se3 object
        H_joint = pin_data_red.oMi[-1].homogeneous # np matrix

        # frame (body) coordinate frames oMf
        pin.updateFramePlacements(pin_model_red, pin_data_red)
        H_body = pin_data_red.oMf[-1].homogeneous

        # joint jacobians
        pin.computeJointJacobians(pin_model_red, pin_data_red, q)
        J_joint = pin_data_red.J[-1]
        
        # inverse dynamics
        pin_tau = pin.rnea(pin_model_red, pin_data_red, q, v, a)
        mj_tau = (mj_data.qfrc_passive + mj_data.qfrc_actuator + mj_data.qfrc_applied)[0:nq]
        print(f"delta tau: {np.linalg.norm(mj_tau - pin_tau)}", end='\n')  # '\033[F' moves the cursor one line up

        # forward dynamics
        acc_pin = pin.aba(pin_model_red, pin_data_red, q, v, pin_tau)
        acc_mj = mj_data.qacc[0:nq]
        print(f"delta acc: {np.linalg.norm(acc_mj - acc_pin)}", end='\033[F')  # '\r' brings the cursor to the beginning of the line

        # Mass matrix
        M_pin = pin.crba(pin_model_red, pin_data_red, q)

        # Corriolis and gravity
        nle_pin = pin.nonLinearEffects(pin_model_red, pin_data_red, q, v)

        # Corriolis matrix
        C = pin.computeCoriolisMatrix(pin_model_red, pin_data_red, q, v)

        # Gravity
        g = pin.computeGeneralizedGravity(pin_model_red, pin_data_red, q)

        ### END PINOCCHINO ###




        # Calculate time remaining until next frame
        time_until_next_frame = frame_time - (time.time() - last_frame_time)
        if time_until_next_frame < 0:
            # If time until next frame is negative, reset last frame time and synchronize with viewer
            last_frame_time = time.time()
            viewer.sync()  # Pick up changes to the physics state, apply perturbations, update options from GUI

        # Sleep to maintain the desired timestep
        time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
