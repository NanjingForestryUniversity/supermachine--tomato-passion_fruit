import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import win32pipe
import win32file
import struct
from PIL import Image
import io

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
            with open(image_path, 'rb') as f:
                img_data = f.read()

            try:
                win32file.WriteFile(self.rgb_send, len(img_data).to_bytes(4, byteorder='big'))
                win32file.WriteFile(self.rgb_send, img_data)
            except Exception as e:
                print(f"数据发送失败. 错误原因: {e}")

        if spec_files:
            spec_file = spec_files[0]
            with open(spec_file, 'rb') as f:
                spec_data = f.read()

            try:
                win32file.WriteFile(self.spec_send, len(spec_data).to_bytes(4, byteorder='big'))
                print(f"发送的光谱数据长度: {len(spec_data)}")
                win32file.WriteFile(self.spec_send, spec_data)
                print(f'发送的光谱数据长度: {len(spec_data)}')
            except Exception as e:
                print(f"数据发送失败. 错误原因: {e}")

        self.receive_result()

    def receive_result(self):
        try:
            # 读取结果数据
            long_axis = int.from_bytes(win32file.ReadFile(self.rgb_receive, 2)[1], byteorder='big')
            short_axis = int.from_bytes(win32file.ReadFile(self.rgb_receive, 2)[1], byteorder='big')
            defect_num = int.from_bytes(win32file.ReadFile(self.rgb_receive, 2)[1], byteorder='big')
            total_defect_area = int.from_bytes(win32file.ReadFile(self.rgb_receive, 4)[1], byteorder='big')
            len_img = int.from_bytes(win32file.ReadFile(self.rgb_receive, 4)[1], byteorder='big')
            img_data = win32file.ReadFile(self.rgb_receive, len_img)[1]

            print(f"长径: {long_axis}, 短径: {short_axis}, 缺陷个数: {defect_num}, 缺陷面积: {total_defect_area}")

            # 显示结果图像
            image = Image.open(io.BytesIO(img_data))
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
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    main_window.open_file_dialog()
    sys.exit(app.exec_())