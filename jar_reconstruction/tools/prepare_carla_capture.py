"""显式整理采集索引；可记录人工确认的传感器对照证据，不修改原始图像。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.carla_capture import main

if __name__ == "__main__":
    main()
