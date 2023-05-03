# -*- coding:utf-8 -*-
# Author: lqxu

import math 


def absmax_quantize(params: list[float]):
    absmax = max([abs(param) for param in params])
    
    factor = round(127 / absmax, 1)
    
    results = [param * factor for param in params]
    
    results = [int(round(result, 0)) for result in results]
    
    return results, factor


def absmax_de_quantize(params: list[int], factor: float):
    results = [param / factor for param in params]
    
    results = [round(result, 1) for result in results]
    
    return results 


if __name__ == "__main__":
    # o_params = [1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4]
    o_params = [-0.10, -0.23,  0.08, -0.38, -0.28, -0.29, -2.11,  0.34, -0.53, -67.0]
    
    n_params, factor = absmax_quantize(o_params)
    
    print(o_params)
    print(factor)
    print(n_params)
    print(absmax_de_quantize(n_params, factor))
    