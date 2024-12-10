# import numpy as np
#
# # 读取 numpy 文件
# file_path = "../anchors_palm.npy"  # 替换为你的文件路径
# variable_name = "anchors_palm"
#
# matrix = np.load(file_path)
#
# # 检查是否是二维矩阵
# if len(matrix.shape) != 2:
#     raise ValueError("The input .npy file is not a 2D matrix.")
#
# # 生成 C++ 数组
# rows, cols = matrix.shape
# header_file = "matrix_data.h"
#
# with open(header_file, "w") as f:
#     f.write(f"#pragma once\n\n")
#     f.write(f"static const int {variable_name}_rows = {rows};\n")
#     f.write(f"static const int {variable_name}_cols = {cols};\n")
#     f.write(f"static const float {variable_name}[{rows}][{cols}] = {{\n")
#     for row in matrix:
#         row_data = ", ".join(f"{val:.6f}" for val in row)  # 保留小数点后 6 位
#         f.write(f"    {{ {row_data} }},\n")
#     f.write("};\n")
# print(f"Header file '{header_file}' generated.")



import numpy as np

# 读取 numpy 文件
file_path = "../anchors_palm.npy"  # 替换为你的文件路径
variable_name = "anchors_palm"

# 加载数据
matrix = np.load(file_path)

# 检查是否是二维矩阵
if len(matrix.shape) != 2:
    raise ValueError("The input .npy file is not a 2D matrix.")

# 将二维矩阵转化为一维数组
flattened_array = matrix.flatten()

# 获取一维数组的长度
array_length = len(flattened_array)

# 生成 C++ 头文件
header_file = "matrix_data.h"

with open(header_file, "w") as f:
    f.write(f"#pragma once\n\n")
    f.write(f"static const int {variable_name}_length = {array_length};\n")
    f.write(f"static const float {variable_name}[{array_length}] = {{\n")
    for i, value in enumerate(flattened_array):
        f.write(f"    {value:.6f}")
        if i < array_length - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n")

print(f"Header file '{header_file}' generated.")

