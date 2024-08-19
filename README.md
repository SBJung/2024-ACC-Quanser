# 2024 ACC Self-Driving Car Student Competition

## Member

Kyungpoook National University, South Korea

**Team Name:** VOICE  
**Members:** Suyong Park, Jiwoo Oh, Chanhyuk Lee, Junghyo Kim, Hyeonjeong Kim, Sebin Jung, Ginyeong Yang, Dongryeol Won

## Qualifying Match Video
[Youtube Link](https://youtu.be/fgEqlnd90-g)

## User Guide

- Run `Autonomous_Drive.py` (No need to run any other file. All libraries and modules contained inside. Please place all files in the same folder and run.)
- Commented out the following part of the `Setup_Competition.py` to prevent real-time running with a spawned virtual model:
    - line 45: `# QLabsRealTime().terminate_all_real_time_models()`
- If you wish to see obstacle avoidance:
    - Uncomment line 169 - 170 in `Autonomous_Drive.py`
- If you want to watch the real-time video of recognition:
    - Uncomment line 53, 54, 176 in `Autonomous_Drive.py`

## Key Points

- Control logic based on Pure Pursuit
- Adaptive speed profile implemented based on steering angle
- Avoidance logic based on LiDAR scan
- Recognition logic based on YOLOv7

## PC Specification

- i7 @2.50GHz CPU
- 32.0GB Memory
- Python 3.11.4
- Windows 10
