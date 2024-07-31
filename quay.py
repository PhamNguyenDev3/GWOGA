# import pandas as pd

# # Đọc dữ liệu từ file CSV
# f1_data = pd.read_csv('./dim50/comparison_function_F1.csv')
# f2_data = pd.read_csv('./dim50/comparison_function_F2.csv')
# f3_data = pd.read_csv('./dim50/comparison_function_F3.csv')
# f4_data = pd.read_csv('./dim50/comparison_function_F4.csv')
# f5_data = pd.read_csv('./dim50/comparison_function_F5.csv')
# f6_data = pd.read_csv('./dim50/comparison_function_F6.csv')
# f7_data = pd.read_csv('./dim50/comparison_function_F7.csv')
# f8_data = pd.read_csv('./dim50/comparison_function_F8.csv')

# # Chuyển đổi dữ liệu sang dạng ngang
# f1_pivot = f1_data.T
# f2_pivot = f2_data.T
# f3_pivot = f3_data.T
# f4_pivot = f4_data.T
# f5_pivot = f5_data.T
# f6_pivot = f6_data.T
# f7_pivot = f7_data.T
# f8_pivot = f8_data.T

# # Hiển thị dữ liệu đã chuyển đổi
# print("Function Test F1")
# print(f1_pivot)

# print("\nFunction Test F2")
# print(f2_pivot)

# print("\nFunction Test F3")
# print(f3_pivot)

# print("\nFunction Test F4")
# print(f4_pivot)

# print("\nFunction Test F5")
# print(f5_pivot)

# print("\nFunction Test F6")
# print(f6_pivot)

# print("\nFunction Test F7")
# print(f7_pivot)

# print("\nFunction Test F8")
# print(f8_pivot)
import pandas as pd

# Đọc dữ liệu từ file CSV
f1_data = pd.read_csv('./comparison_function_F1_10.csv')
f2_data = pd.read_csv('./comparison_function_F2_10.csv')
f3_data = pd.read_csv('./comparison_function_F3_10.csv')
f4_data = pd.read_csv('./comparison_function_F4_10.csv')
f5_data = pd.read_csv('./comparison_function_F5_10.csv')
f6_data = pd.read_csv('./comparison_function_F6_10.csv')
f7_data = pd.read_csv('./comparison_function_F7_10.csv')
f8_data = pd.read_csv('./comparison_function_F8_10.csv')
f9_data = pd.read_csv('./comparison_function_F9_10.csv')


# Chuyển đổi dữ liệu sang dạng ngang
f1_pivot = f1_data.T
f2_pivot = f2_data.T
f3_pivot = f3_data.T
f4_pivot = f4_data.T
f5_pivot = f5_data.T
f6_pivot = f6_data.T
f7_pivot = f7_data.T
f8_pivot = f8_data.T
f9_pivot = f9_data.T


# Đổi tên cột đầu tiên thành tên các hàm
f1_pivot.columns = f1_pivot.iloc[0]
f2_pivot.columns = f2_pivot.iloc[0]
f3_pivot.columns = f3_pivot.iloc[0]
f4_pivot.columns = f4_pivot.iloc[0]
f5_pivot.columns = f5_pivot.iloc[0]
f6_pivot.columns = f6_pivot.iloc[0]
f7_pivot.columns = f7_pivot.iloc[0]
f8_pivot.columns = f8_pivot.iloc[0]
f9_pivot.columns = f9_pivot.iloc[0]


# Bỏ hàng đầu tiên sau khi đổi tên cột
f1_pivot = f1_pivot[1:]
f2_pivot = f2_pivot[1:]
f3_pivot = f3_pivot[1:]
f4_pivot = f4_pivot[1:]
f5_pivot = f5_pivot[1:]
f6_pivot = f6_pivot[1:]
f7_pivot = f7_pivot[1:]
f8_pivot = f8_pivot[1:]
f9_pivot = f9_pivot[1:]


# Tạo file CSV tổng hợp
with open('combined_functions_10.csv', 'w') as f:
    f.write("Function Test F1\n")
    f1_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F2\n")
    f2_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F3\n")
    f3_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F4\n")
    f4_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F5\n")
    f5_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F6\n")
    f6_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F7\n")
    f7_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F8\n")
    f8_pivot.to_csv(f, header=True)
    f.write("\nFunction Test F9\n")
    f9_pivot.to_csv(f, header=True)

print("Dữ liệu đã được ghi vào file 'combined_functions_10.csv'")
