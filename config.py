"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models') 

DRIVE_LOOP_HZ = 20      
MAX_LOOPS = None        

CAMERA_TYPE = "MOCK"  
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3        
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_INDEX = 0  
CSIC_CAM_GSTREAMER_FLIP_PARM = 0
BGR2RGB = False  
SHOW_PILOT_IMAGE = False  

PCA9685_I2C_ADDR = 0x40    
PCA9685_I2C_BUSNUM = None  

USE_SSD1306_128_32 = False    
SSD1306_128_32_I2C_ROTATION = 0
SSD1306_RESOLUTION = 1

DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:40.1",  
    "PWM_STEERING_SCALE": 1.0,              
    "PWM_STEERING_INVERTED": False,        
    "PWM_THROTTLE_PIN": "PCA9685.1:40.0",  
    "PWM_THROTTLE_SCALE": 1.0,              
    "PWM_THROTTLE_INVERTED": False,        
    "STEERING_LEFT_PWM": 460,              
    "STEERING_RIGHT_PWM": 290,              
    "THROTTLE_FORWARD_PWM": 500,            
    "THROTTLE_STOPPED_PWM": 370,            
    "THROTTLE_REVERSE_PWM": 220,            
}

STEERING_CHANNEL = 1            
STEERING_LEFT_PWM = 460        
STEERING_RIGHT_PWM = 290        
THROTTLE_CHANNEL = 0            
THROTTLE_FORWARD_PWM = 500      
THROTTLE_STOPPED_PWM = 370      
THROTTLE_REVERSE_PWM = 220      

STEERING_PWM_PIN = 13          
STEERING_PWM_FREQ = 50          
STEERING_PWM_INVERTED = False  
THROTTLE_PWM_PIN = 18          
THROTTLE_PWM_FREQ = 50          
THROTTLE_PWM_INVERTED = False  

SERVO_HBRIDGE_2PIN = {
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",      
    "PWM_STEERING_SCALE": 1.0,        
    "PWM_STEERING_INVERTED": False,  
    "STEERING_LEFT_PWM": 460,        
    "STEERING_RIGHT_PWM": 290,        
}

SERVO_HBRIDGE_3PIN = {
    "FWD_PIN": "RPI_GPIO.BOARD.18",  
    "BWD_PIN": "RPI_GPIO.BOARD.16",  
    "DUTY_PIN": "RPI_GPIO.BOARD.35",  
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",  
    "PWM_STEERING_SCALE": 1.0,        
    "PWM_STEERING_INVERTED": False,  
    "STEERING_LEFT_PWM": 460,        
    "STEERING_RIGHT_PWM": 290,        
}

HBRIDGE_PIN_FWD = 18      
HBRIDGE_PIN_BWD = 16      
STEERING_CHANNEL = 0      
STEERING_LEFT_PWM = 460    
STEERING_RIGHT_PWM = 290  

VESC_MAX_SPEED_PERCENT =.2  
VESC_SERIAL_PORT= "/dev/ttyACM0"
VESC_HAS_SENSOR= True
VESC_START_HEARTBEAT= True
VESC_BAUDRATE= 115200
VESC_TIMEOUT= 0.05
VESC_STEERING_SCALE= 0.5
VESC_STEERING_OFFSET = 0.5

DC_STEER_THROTTLE = {
    "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",  
    "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    
}

DC_TWO_WHEEL = {
    "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  
    "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  
    "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",
    "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",
}

DC_TWO_WHEEL_L298N = {
    "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        
    "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        
    "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    

    "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",      
    "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",      
    "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",  
}

HAVE_ODOM = False                  
ENCODER_TYPE = 'GPIO'            
MM_PER_TICK = 12.7625              
ODOM_PIN = 13                        
ODOM_DEBUG = False                  

USE_LIDAR = False
LIDAR_TYPE = 'RP'
LIDAR_LOWER_LIMIT = 90
LIDAR_UPPER_LIMIT = 270

HAVE_TFMINI = False
TFMINI_SERIAL_PORT = "/dev/serial0"

DEFAULT_AI_FRAMEWORK = 'tensorflow'

DEFAULT_MODEL_TYPE = 'linear'
BATCH_SIZE = 128                
TRAIN_TEST_SPLIT = 0.8          
MAX_EPOCHS = 100                
SHOW_PLOT = True                
VERBOSE_TRAIN = True            
USE_EARLY_STOP = True          
EARLY_STOP_PATIENCE = 5        
MIN_DELTA = .0005              
PRINT_MODEL_SUMMARY = True      
OPTIMIZER = None                
LEARNING_RATE = 0.001          
LEARNING_RATE_DECAY = 0.0      
SEND_BEST_MODEL_TO_PI = False  
CREATE_TF_LITE = True          
CREATE_TENSOR_RT = False        
SAVE_MODEL_AS_H5 = False        
CACHE_POLICY = 'ARRAY'          

PRUNE_CNN = False              
PRUNE_PERCENT_TARGET = 75      
PRUNE_PERCENT_PER_ITERATION = 20
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2
PRUNE_EVAL_PERCENT_OF_DATASET = .05  

AUGMENTATIONS = []         
TRANSFORMATIONS = []       
POST_TRANSFORMATIONS = []  

AUG_BRIGHTNESS_RANGE = 0.2
AUG_BLUR_RANGE = (0, 3)

ROI_CROP_TOP = 45              
ROI_CROP_BOTTOM = 0            
ROI_CROP_RIGHT = 0              
ROI_CROP_LEFT = 0              

ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = 160
ROI_TRAPEZE_UL = 20
ROI_TRAPEZE_UR = 140
ROI_TRAPEZE_MIN_Y = 60
ROI_TRAPEZE_MAX_Y = 120

CANNY_LOW_THRESHOLD = 60    
CANNY_HIGH_THRESHOLD = 110  
CANNY_APERTURE = 3          

BLUR_KERNEL = 5        
BLUR_KERNEL_Y = None  
BLUR_GAUSSIAN = True  

RESIZE_WIDTH = 160    
RESIZE_HEIGHT = 120    

SCALE_WIDTH = 1.0      
SCALE_HEIGHT = None    

FREEZE_LAYERS = False              
NUM_LAST_LAYERS_TO_TRAIN = 7        

WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  
WEB_INIT_MODE = "user"              

USE_JOYSTICK_AS_DEFAULT = False      
JOYSTICK_MAX_THROTTLE = 0.5        
JOYSTICK_STEERING_SCALE = 1.0      
AUTO_RECORD_ON_THROTTLE = True      
CONTROLLER_TYPE = 'xbox'            
USE_NETWORKED_JS = False            
NETWORK_JS_SERVER_IP = None        
JOYSTICK_DEADZONE = 0.01            
JOYSTICK_THROTTLE_DIR = -1.0        
USE_FPV = False                    
JOYSTICK_DEVICE_FILE = "/dev/input/js0"

MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

SEQUENCE_LENGTH = 3            

HAVE_IMU = False                
IMU_SENSOR = 'mpu6050'          
IMU_ADDRESS = 0x68              
IMU_DLP_CONFIG = 0              

HAVE_SOMBRERO = False          

STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 20
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_MID = 1500        
PIGPIO_MAX_FORWARD = 2000
PIGPIO_STOPPED_PWM = 1500
PIGPIO_MAX_REVERSE = 1000
PIGPIO_SHOW_STEERING_VALUE = False
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.025


MM1_STEERING_MID = 1500
MM1_MAX_FORWARD = 2000
MM1_STOPPED_PWM = 1500
MM1_MAX_REVERSE = 1000
MM1_SHOW_STEERING_VALUE = False
MM1_SERIAL_PORT = '/dev/ttyS0'

HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(message)s'

HAVE_MQTT_TELEMETRY = False
TELEMETRY_DONKEY_NAME = 'my_robot1234'
TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
TELEMETRY_MQTT_JSON_ENABLE = False
TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
TELEMETRY_MQTT_BROKER_PORT = 1883
TELEMETRY_PUBLISH_PERIOD = 1
TELEMETRY_LOGGING_ENABLE = True
TELEMETRY_LOGGING_LEVEL = 'INFO'
TELEMETRY_LOGGING_FORMAT = '%(message)s'
TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
TELEMETRY_DEFAULT_TYPES = 'float,float'

HAVE_PERFMON = False

RECORD_DURING_AI = False
AUTO_CREATE_NEW_TUB = False

HAVE_RGB_LED = False
LED_INVERT = False

LED_PIN_R = 12
LED_PIN_G = 10
LED_PIN_B = 16

LED_R = 0
LED_G = 0
LED_B = 1

REC_COUNT_ALERT = 1000
REC_COUNT_ALERT_CYC = 15
REC_COUNT_ALERT_BLINK_RATE = 0.4

RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]


MODEL_RELOADED_LED_R = 100
MODEL_RELOADED_LED_G = 0
MODEL_RELOADED_LED_B = 0


TRAIN_BEHAVIORS = False
BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]

TRAIN_LOCALIZER = False
NUM_LOCATIONS = 10
BUTTON_PRESS_NEW_TUB = False

DONKEY_GYM = False
DONKEY_SIM_PATH = "path to sim"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"
GYM_CONF = {
    'exe_path': DONKEY_SIM_PATH,
    'host': SIM_HOST,
    'port': SIM_PORT,
    'env_name': GYM_ENV_NAME,
    'camera_resolution': CAM_RESOLUTION,
    'start_delay': 5.0,
    'max_cte': 8.0,
    'reward_speed_weight': 0.15,
    'reward_cte_weight': 1.0,
    'reward_smooth_steer_weight': 0.05,
    'penalty_collision_heavy': -20.0,
    'penalty_collision_light': -5.0,
    'cte_tolerance': 0.5,
,"body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100}
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"
SIM_ARTIFICIAL_LATENCY = 0

SIM_RECORD_LOCATION = False
SIM_RECORD_GYROACCEL= False
SIM_RECORD_VELOCITY = False
SIM_RECORD_LIDAR = False

PUB_CAMERA_IMAGES = False

AI_LAUNCH_DURATION = 0.0
AI_LAUNCH_THROTTLE = 0.0
AI_LAUNCH_ENABLE_BUTTON = 'R2'
AI_LAUNCH_KEEP_ENABLED = False

AI_THROTTLE_MULT = 1.0

PATH_FILENAME = "donkey_path.pkl"
PATH_SCALE = 5.0
PATH_OFFSET = (0, 0)
PATH_MIN_DIST = 0.3
PID_P = -10.0
PID_I = 0.000
PID_D = -0.2
PID_THROTTLE = 0.2
USE_CONSTANT_THROTTLE = False
SAVE_PATH_BTN = "cross"
RESET_ORIGIN_BTN = "triangle"

REALSENSE_D435_RGB = True
REALSENSE_D435_DEPTH = True
REALSENSE_D435_IMU = False
REALSENSE_D435_ID = None

STOP_SIGN_DETECTOR = False
STOP_SIGN_MIN_SCORE = 0.2
STOP_SIGN_SHOW_BOUNDING_BOX = True
STOP_SIGN_MAX_REVERSE_COUNT = 10
STOP_SIGN_REVERSE_THROTTLE = -0.5

SHOW_FPS = False
FPS_DEBUG_INTERVAL = 10

PI_USERNAME = "pi"
PI_HOSTNAME = "donkeypi.local"
SIM_TYPE = 'unity'
WEB_SERVER_ENABLED = True
WEB_SERVER_PORT = 8887