"""只读比较同帧雷达与深度，结果用于确认数据含义，不是网络精度。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.carla_depth_audit import main

if __name__ == "__main__":
    main()
