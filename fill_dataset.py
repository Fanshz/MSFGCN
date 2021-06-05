import xlwt
import xlrd
import random

def write_excel():
  # 2. 创建Excel工作薄
  myWorkbook = xlwt.Workbook()
  # 3. 添加Excel工作表
  mySheet = myWorkbook.add_sheet('V')
  workbook = xlrd.open_workbook(r'D:\Project\MS-GCN_edit\MSGCN\dataset\V.xls')

  # sheet_name = workbook.sheet_names()[0]

  # 根据sheet索引或者名称获取sheet内容
  sheet = workbook.sheet_by_index(0)  # sheet索引从0开始
  # sheet = workbook.sheet_by_name('Sheet1')

  # print (workboot.sheets()[0])
  # sheet的名称，行数，列数
  # print(sheet.name, sheet.nrows, sheet.ncols)

  # 获取整行和整列的值（数组）
  # rows = sheet.row_values(1)  # 获取第2行内容
  # cols = sheet.col_values(2) # 获取第3列内容
  # print(sheet.cell_value(1, 2))

  # 4. 写入数据
  # myStyle = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')#数据格式
  for i in range(12672):
      for j in range(228):
          if i<2976 and j<130:
              mySheet.write(i, j, sheet.cell_value(i, j))
              pass
          else:
              # mySheet.write(i, j, sheet.cell_value(i%2976, j%130)+ random.uniform(0, 5))
              mySheet.write(i, j, 0)
  #5. 保存
  myWorkbook.save('D:\Project\MS-GCN_edit\MSGCN\dataset\V_228.xls')
if __name__ == '__main__':
    # 写入Excel
    write_excel()
    print ('data has made over！')