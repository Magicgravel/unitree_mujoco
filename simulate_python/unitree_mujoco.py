import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import argparse
import os
import yaml

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

from motion.motion import MotionController, MoveController

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.USE_KEYBOARD:
        unitree.keyboard = True
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)

def MotionThread():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "motion", "config.yaml")
    with open(config_file, 'r') as f:
        print(f"Loading config from {config_file}")
        cfg = yaml.safe_load(f)
    
    motion_controller = MotionController(cfg)
    
    if cfg.get("input_mode") == "api":
        move_controller = MoveController(motion_controller)
    
    while viewer.is_running():
        loop_start_time = time.time()
        
        motion_controller.step()
        
        elapsed = time.time() - loop_start_time
        sleep_time = motion_controller.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Warning: Loop is running behind by {-sleep_time:.3f} seconds.")
    
    motion_controller.stop()


def AudioThread():
    """
    Run a simulated G1 audio RPC service ("voice").

    It receives PlayStream requests sent by t_g1_play_wav.py (16kHz mono PCM)
    and plays them out through the local speaker, so the simulator behaves like
    the real robot's audio service.
    """
    import json
    import subprocess
    from unitree_sdk2py.rpc.server import Server
    from unitree_sdk2py.rpc.internal import (
        RPC_OK,
        RPC_ERR_SERVER_API_PARAMETER,
        RPC_ERR_SERVER_INTERNAL,
    )
    from unitree_sdk2py.g1.audio.g1_audio_api import (
        AUDIO_SERVICE_NAME,
        AUDIO_API_VERSION,
        ROBOT_API_ID_AUDIO_START_PLAY,
        ROBOT_API_ID_AUDIO_STOP_PLAY,
        ROBOT_API_ID_AUDIO_GET_VOLUME,
        ROBOT_API_ID_AUDIO_SET_VOLUME,
    )

    class AudioServer(Server):
        def __init__(self):
            super().__init__(AUDIO_SERVICE_NAME)
            self._volume = 80
            self._player = None
            self._lock = threading.Lock()

        def Init(self):
            self._SetApiVersion(AUDIO_API_VERSION)
            self._RegistBinaryHandler(ROBOT_API_ID_AUDIO_START_PLAY, self._OnStartPlay, False)
            self._RegistHandler(ROBOT_API_ID_AUDIO_STOP_PLAY, self._OnStopPlay, False)
            self._RegistHandler(ROBOT_API_ID_AUDIO_GET_VOLUME, self._OnGetVolume, False)
            self._RegistHandler(ROBOT_API_ID_AUDIO_SET_VOLUME, self._OnSetVolume, False)

        def _OnStartPlay(self, pcm_binary):
            with self._lock:
                if self._player is None or self._player.poll() is not None:
                    self._player = subprocess.Popen(
                        ["aplay", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1"],
                        stdin=subprocess.PIPE,
                    )
                try:
                    self._player.stdin.write(bytes(pcm_binary))
                    self._player.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    print(f"[AudioServer] play PCM failed: {e}")
                    self._player = None
                    return RPC_ERR_SERVER_INTERNAL, []
            return RPC_OK, []

        def _OnStopPlay(self, parameter):
            with self._lock:
                if self._player is not None:
                    if self._player.poll() is None:
                        try:
                            self._player.stdin.close()
                        except OSError:
                            pass
                        try:
                            self._player.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            self._player.kill()
                            self._player.wait()
                    self._player = None
            print("[AudioServer] playback stopped.")
            return RPC_OK, ""

        def _OnGetVolume(self, parameter):
            return RPC_OK, json.dumps({"volume": self._volume})

        def _OnSetVolume(self, parameter):
            try:
                p = json.loads(parameter)
                self._volume = int(p.get("volume", self._volume))
            except (ValueError, TypeError):
                return RPC_ERR_SERVER_API_PARAMETER, ""
            print(f"[AudioServer] volume set to {self._volume}")
            return RPC_OK, ""

    audio_server = AudioServer()
    audio_server.Init()
    audio_server.Start()
    print("[AudioServer] started, waiting for PlayStream requests...")

    while viewer.is_running():
        time.sleep(0.5)

    audio_server._OnStopPlay("{}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--audio', action='store_true', help='Enable audio')
    args = parser.parse_args()
    print(f'args: {args}')
    
    # Initialize DDS before starting any threads that use it
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    
    if args.motion:
        motion_thread = Thread(target=MotionThread)
        motion_thread.start()
    if args.audio:
        audio_thread = Thread(target=AudioThread)
        audio_thread.start()

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
