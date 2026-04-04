class ColorText:
    # 定义颜色代码
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

    # 定义样式代码
    BOLD = "\033[1m"        # 加粗
    ITALIC = "\033[3m"      # 斜体
    UNDERLINE = "\033[4m"   # 下划线
    REVERSE = "\033[7m"     # 反显
    BLINK = "\033[5m"       # 闪烁
    BLINK_FAST = "\033[6m"  # 快速闪烁
    HIDDEN = "\033[8m"      # 隐藏
    STRIKE = "\033[9m"      # 删除线

    @staticmethod
    def color_text(text, color, style=None):
        """
        给文本添加颜色和样式
        :param text: 要显示的文本
        :param color: 文本颜色
        :param style: 文本样式（可选）
        :return: 添加颜色和样式的文本
        """
        if style:
            return f"{style}{color}{text}{ColorText.RESET}"
        return f"{color}{text}{ColorText.RESET}"

    @staticmethod
    def print(text, color, style=None):
        """
        打印带有颜色和样式的文本
        :param text: 要打印的文本
        :param color: 文本颜色
        :param style: 文本样式（可选）
        """
        print(ColorText.color_text(text, color, style))


# 示例用法
if __name__ == "__main__":
    ColorText.print("这是紫色文字", ColorText.PURPLE, ColorText.ITALIC)
    ColorText.print("这是反显白字文字", ColorText.WHITE, ColorText.REVERSE)
    ColorText.print("这是加粗的绿色文字", ColorText.GREEN, ColorText.BOLD)
    ColorText.print("这是带有下划线的蓝色文字", ColorText.BLUE, ColorText.UNDERLINE)
    ColorText.print("这是反显的黄色文字", ColorText.YELLOW, ColorText.REVERSE)
    print("ceshi")