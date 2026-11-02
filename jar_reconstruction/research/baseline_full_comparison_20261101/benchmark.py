"""复用同卡测速器；随机模型按原生推理运行，不计精度协议的哈希开销。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v5_comparison_20261101.evaluate import Models

if __name__ == "__main__":
    if "--four" in sys.argv:
        sys.argv.remove("--four")
        from research.v5_comparison_20261101 import speed as benchmark
    else:
        from tools import benchmark_public_speed as benchmark
    benchmark.Models = Models
    benchmark.main()
