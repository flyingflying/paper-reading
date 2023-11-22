# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM 

from itertools import pairwise
from collections import Counter, deque


def _get_unused_byte(byte_set):
    for idx, byte in zip(range(255, -1, -1), sorted(byte_set, reverse=True)):
        if idx != byte:
            return idx
    return idx - 1


def compress_block(block: bytes, byte_set: set[int], min_occurrence: int = 3):    
    rp_dict = dict()

    while len(byte_set) < 256:

        # https://docs.python.org/3/library/itertools.html 
        bp_counter = Counter(pairwise(block))
        bp, occurrence = bp_counter.most_common(1)[0]

        if occurrence <= min_occurrence:
            break

        unused_byte = _get_unused_byte(byte_set)
        byte_set.add(unused_byte)

        rp_dict[unused_byte] = bp
        block = block.replace(bytes(bp), bytes([unused_byte, ])) 

    return block, rp_dict


def expand_block(block: bytes, rp_dict: dict[bytes, bytes]):
    results = []
    stack = deque()  # LIFO

    for sbyte in block:

        stack.appendleft(sbyte)
        
        while len(stack) != 0:
            sbyte = stack.popleft()
            
            if sbyte not in rp_dict:
                results.append(sbyte)
                continue
            
            stack.extendleft(rp_dict[sbyte][::-1])
    
    return bytes(results)


if __name__ == "__main__":
    BLOCK_SIZE = 2000
    MAX_KINDS = 200
    MIN_OCCURRENCE = 3

    with open(__file__, "rb") as reader_:
        while True:

            buffer_ = []
            byte_set_ = set()
            
            for _ in range(BLOCK_SIZE):
                sbyte_ = reader_.read(1)

                # python 中没有 EOF 的概念, 如果文件读完了, 则返回空 bytes 对象
                if len(sbyte_) == 0:
                    break 

                sbyte_ = sbyte_[0]
                buffer_.append(sbyte_)
                byte_set_.add(sbyte_)
                
                if len(byte_set_) > MAX_KINDS:
                    # byte 一共只有 256 种
                    # 每一个 block 中 byte 的种类不能超过 MAX_KINDS, 否则就没有 byte 可以替换了
                    break 

            if len(buffer_) == 0:
                break 

            block_ = bytes(buffer_)
            compressed_block_, rp_dict_ = compress_block(block_, byte_set_, MIN_OCCURRENCE)
            # 需要将 compressed_block_ 和 rp_dict 存到压缩后的文件中

            # 解码时, 需要分段读取后解码
            recovered_block_ = expand_block(compressed_block_, rp_dict_)
            
            print(block_ == recovered_block_)
    
    pass
