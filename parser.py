from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading




client = RemoteAPIClient(port=23000)
sim = client.require("sim")

drone = sim.getObject('/Quadcopter')
droneTarget= sim.getObject('/target')

def start():
    sim.setObjectPosition(droneTarget,[0,0,0])
        
    sim.setObjectOrientation(droneTarget,[0,0,0])

def drone_move(target_pos,target_yaw):
    
    sim.setObjectPosition(droneTarget,target_pos)
    euler = list(sim.alphaBetaGammaToYawPitchRoll(*sim.getObjectOrientation(drone)) )
    euler[0]=target_yaw
    sim.setObjectOrientation(droneTarget,[0,0,target_yaw])