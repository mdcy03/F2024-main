from cflib.crazyflie import Crazyflie
import cflib.crtp
import time

# Initialize the low-level drivers
cflib.crtp.init_drivers()

uri = "radio://0/03/2M"

def simple_takeoff(cf):
    print("Taking off...")
    #Send hover commands for 2 seconds to take off
    for _ in range(20):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.5)  # Thrust to 0.5m
        time.sleep(0.1)

    print("Hovering...")
    # Maintain hover for 3 seconds
    for _ in range(30): 
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.5)
        time.sleep(0.1)

    print("Landing...")
    #Gradually decrease thrust to land
    for i in range(20):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.5 - (i * 0.025))
        time.sleep(0.1)

    #stop all motors
    cf.commander.send_stop_setpoint()

def connected_callback(link_uri):
    print(f"Connected to {link_uri}")
    simple_takeoff(cf)
    cf.close_link()


cf = Crazyflie(rw_cache="./cache")
cf.connected.add_callback(connected_callback)

print("Connecting to Crazyflie...")
cf.open_link(uri)
