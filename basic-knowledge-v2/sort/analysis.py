# -*- coding:utf-8 -*-
# Author: lqxu


def selection_sort(array: list):  # 选择排序

    array = list(array)  # 选择排序属于 inplace 操作

    array_length = len(array)

    def argmin(start_idx):
        min_idx, min_value = start_idx, array[start_idx]

        for cur_idx in range(start_idx + 1, array_length):
            cur_value = array[cur_idx]
            if cur_value < min_value:
                min_idx, min_value = cur_idx, cur_value

        return min_idx

    def swap(idx1, idx2):
        array[idx1], array[idx2] = array[idx2], array[idx1]

    for idx in range(array_length - 1):
        selected_idx = argmin(idx)  # 每一次 "选择" 出 未排序部分 的最小值

        swap(idx, selected_idx)  # 无条件交换, 不稳定

    return array


def bubble_sort(array: list):  # 冒泡排序, 基于 "交换" 的
    
    array = list(array)
    
    array_length = len(array)
    
    def swap(idx1, idx2):
        array[idx1], array[idx2] = array[idx2], array[idx1]

    end_idx = array_length

    for _ in range(array_length - 1):  # 对于 冒泡排序 来说, 最多迭代 array_length - 1 次
        last_swap_idx = 0
        
        for idx in range(1, end_idx):
            # 如果右边的元素比左边的小, 就交换
            if array[idx] < array[idx - 1]:  # 有条件交换, 稳定
                swap(idx, idx - 1)
                last_swap_idx = idx
        
        # 对于 冒泡排序 来说, 在 last_swap_idx 之后的序列应该都是有序的
        # 那么我们下一次迭代时, 只需要迭代 last_swap_idx 之前的部分即可
        if last_swap_idx < 2:
            break
        end_idx = last_swap_idx
    
    return array


def insertion_sort(array: list):
    
    array = list(array)
    
    array_length = len(array)
    
    def find_insert_idx(cur_idx_):
        """
        数组在 0 到 cur_idx - 1 的部分是有序的, 我们需要找到 cur_value 在 0 到 cur_idx 部分的正确位置
        
        一种方式是 从左到右 遍历 0 到 cur_idx - 1 部分的数组, 找到第一个比 cur_value 大的值, 取索引作为 insert_idx
        另一种方式是 从右到左 遍历 0 到 cur_idx - 1 部分的数组, 找到第一个比 cur_value 小的值, 取索引加一作为 insert_idx
        """
        cur_value = array[cur_idx_]

        for idx in range(0, cur_idx_):
            if cur_value < array[idx]:
                return idx
        return cur_idx_  # 没有找到, 就是当前位置
    
    def swap(idx1, idx2):
        array[idx1], array[idx2] = array[idx2], array[idx1]
    
    def insert(insert_idx_, cur_idx_):
        """ 对于数组来说, insert 操作涉及到大量的 swap 操作, 非常恐怖 """
        for idx in range(cur_idx_, insert_idx_, -1):  # 从右往左遍历
            swap(idx, idx-1)
    
    for cur_idx in range(1, array_length):
        insert_idx = find_insert_idx(cur_idx)
        insert(insert_idx, cur_idx)
    
    return array


if __name__ == "__main__":
    test_case = [6, 5, 4, 3, 2, 100]
    
    print(selection_sort(test_case))
    print(bubble_sort(test_case))
    print(insertion_sort(test_case))
    