import sys


cn_punc = '，。；：？！（）～｜'  #
def q2b(uchar, skip_cn_punc=False):
    # 有时，希望保留全角中文标点，例如cn_punc。
    if skip_cn_punc and uchar in cn_punc:
        return uchar
    inside_code = ord(uchar)
    if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
    elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
        inside_code -= 65248
    return chr(inside_code)

def str_q2b(ustring, skip_cn_punc=False):
    """ 全角转半角 """
    return ''.join([q2b(uchar, skip_cn_punc) for uchar in ustring])


class TextProcessor:
    def __init__(self):
        #self.converter = Converter('zh-hans')
        pass

    def clean(self, text):
        """
        :param text:
        :return:
        """
        text = str_q2b(text, skip_cn_punc=True)  # 全角->半角
        #text = self.converter.convert(text)  # 繁体->简体
        return text


