# -*- coding: utf-8 -*-
# @Time    : 2024/6/16 17:13
# @Author  : TG
# @File    : qt_test.py
# @Software: PyCharm

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import win32pipe
import win32file
import struct
from PIL import Image
import io
import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tomato Image Sender")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.rgb_send_name = r'\\.\pipe\rgb_receive'  # 发送数据管道名对应 main.py 的接收数据管道名
        self.rgb_receive_name = r'\\.\pipe\rgb_send'  # 接收数据管道名对应 main.py 的发送数据管道名
        self.spec_send_name = r'\\.\pipe\spec_receive' # 发送数据管道名对应 main.py 的接收数据管道名

        # 连接main.py创建的命名管道
        self.rgb_send = win32file.CreateFile(
            self.rgb_send_name,
            win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )

        self.rgb_receive = win32file.CreateFile(
            self.rgb_receive_name,
            win32file.GENERIC_READ,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )

        self.spec_send = win32file.CreateFile(
            self.spec_send_name,
            win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )

    def send_image_group(self, image_dir):
        rgb_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.bmp'))][:5]
        spec_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.raw')][:1]

        for image_path in rgb_files:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.asarray(img, dtype=np.uint8)


            try:
                # win32file.WriteFile(self.rgb_send, len(img_data).to_bytes(4, byteorder='big'))
                height = img.shape[0]
                width = img.shape[1]
                height = height.to_bytes(2, byteorder='big')
                width = width.to_bytes(2, byteorder='big')
                img_data = img.tobytes()
                length = (len(img_data) + 6).to_bytes(4, byteorder='big')
                cmd = 'PF'
                data_send = length + cmd.upper().encode('ascii') + height + width + img_data
                win32file.WriteFile(self.rgb_send, data_send)
                print(f'发送的图像数据长度: {len(data_send)}')
            except Exception as e:
                print(f"数据发送失败. 错误原因: {e}")

        if spec_files:
            spec_file = spec_files[0]
            with open(spec_file, 'rb') as f:
                spec_data = f.read()

            try:
                # win32file.WriteFile(self.spec_send, len(spec_data).to_bytes(4, byteorder='big'))
                # print(f"发送的光谱数据长度: {len(spec_data)}")
                heigth = 30
                weight = 30
                bands = 224
                heigth = heigth.to_bytes(2, byteorder='big')
                weight = weight.to_bytes(2, byteorder='big')
                bands = bands.to_bytes(2, byteorder='big')
                length = (len(spec_data)+8).to_bytes(4, byteorder='big')
                cmd = 'PF'
                data_send = length + cmd.upper().encode('ascii') + heigth + weight + bands + spec_data
                win32file.WriteFile(self.spec_send, data_send)
                print(f'发送的光谱数据长度: {len(data_send)}')
                print(f'spec长度: {len(spec_data)}')
            except Exception as e:
                print(f"数据发送失败. 错误原因: {e}")

        self.receive_result()

    def receive_result(self):
        try:
            # 读取结果数据
            # 读取4字节的数据长度信息，并将其转换为整数
            data_length = int.from_bytes(win32file.ReadFile(self.rgb_receive, 4)[1], byteorder='big')
            print(f"应该接收到的数据长度: {data_length}")
            # 根据读取到的数据长度，读取对应长度的数据
            data = win32file.ReadFile(self.rgb_receive, data_length)[1]
            print(f"接收到的数据长度: {len(data)}")
            # 解析数据
            cmd_result = data[:2].decode('ascii').strip().upper()
            brix = int.from_bytes(data[2:4], byteorder='big')
            green_percentage = int.from_bytes(data[4:5], byteorder='big')
            diameter = int.from_bytes(data[5:7], byteorder='big')
            weight = int.from_bytes(data[7:8], byteorder='big')
            defect_num = int.from_bytes(data[8:10], byteorder='big')
            total_defect_area = int.from_bytes(data[10:14], byteorder='big')
            heigth = int.from_bytes(data[14:16], byteorder='big')
            width = int.from_bytes(data[16:18], byteorder='big')
            rp = data[18:]
            print(heigth, width)
            img = np.frombuffer(rp, dtype=np.uint8).reshape(heigth, width, -1)
            print(f"接收到的结果数据: {cmd_result}, {brix}, {green_percentage}, {diameter}, {weight}, {defect_num}, {total_defect_area}, {img.shape}")


            # 显示结果图像
            image = Image.fromarray(img)
            qimage = QImage(image.tobytes(), image.size[0], image.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)

        except Exception as e:
            print(f"数据接收失败. 错误原因: {e}")

    def open_file_dialog(self):
        directory_dialog = QFileDialog()
        directory_dialog.setFileMode(QFileDialog.Directory)
        if directory_dialog.exec_():
            selected_directory = directory_dialog.selectedFiles()[0]
            self.send_image_group(selected_directory)

if __name__ == "__main__":
    '''
    1. 创建Qt应用程序
    2. 创建主窗口
    3. 显示主窗口
    4. 打开文件对话框
    5. 进入Qt事件循环
    '''
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    main_window.open_file_dialog()
    sys.exit(app.exec_())