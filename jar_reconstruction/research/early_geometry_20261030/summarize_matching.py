"""匹配单支路减法实验独立汇总，保留首轮所有原始汇总不覆盖。"""

from summarize import ARMS, run


if __name__ == "__main__":
    run(arms=ARMS + ("matching_geometry", "matching_constant"), prefix="matching_")
