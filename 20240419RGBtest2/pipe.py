import win32file
import win32pipe
import io
from PIL import Image
import time
import logging

def send_data(pipe_send, long_axis, short_axis, defect_num, total_defect_area, rp):


    start_time = time.time()

    # width = rp.shape[0]
    # height = rp.shape[1]
    # print(width, height)
    img_bytes = rp.tobytes()
    # length = len(img_bytes) + 18
    # print(length)
    # length = length.to_bytes(4, byteorder='big')
    # width = width.to_bytes(2, byteorder='big')
    # height = height.to_bytes(2, byteorder='big')
    long_axis = long_axis.to_bytes(2, byteorder='big')
    short_axis = short_axis.to_bytes(2, byteorder='big')
    defect_num = defect_num.to_bytes(2, byteorder='big')
    total_defect_area = int(total_defect_area).to_bytes(4, byteorder='big')
    # cmd_type = 'RIM'
    # result = result.encode('ascii')
    # send_message = b'\xaa' + length + (' ' + cmd_type).upper().encode('ascii') + long_axis + short_axis + defect_num + total_defect_area + width + height + img_bytes + b'\xff\xff\xbb'
    send_message = long_axis + short_axis + defect_num + total_defect_area + img_bytes
    # print(long_axis)
    # print(short_axis)
    # print(defect_num)
    # print(total_defect_area)
    # print(width)
    # print(height)

    try:
        win32file.WriteFile(pipe_send, send_message)
        print('发送成功')
        # print(send_message)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False

    end_time = time.time()
    print(f'发送时间：{end_time - start_time}秒')

    return True


def receive_data(pipe, data_size):
    try:
        # 读取图片数据
        result, img_data = win32file.ReadFile(pipe, data_size, None)
        return img_data
    except Exception as e:
        print(f"Failed to receive data. Error: {e}")
        return None


def create_pipes(pipe_receive_name, pipe_send_name):
    # 打开或创建命名管道
    pipe_receive = win32pipe.CreateNamedPipe(
        pipe_receive_name,
        win32pipe.PIPE_ACCESS_INBOUND,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 80000000, 80000000, 0, None
    )
    pipe_send = win32pipe.CreateNamedPipe(
        pipe_send_name,
        win32pipe.PIPE_ACCESS_OUTBOUND,  # 修改为输出模式
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 80000000, 80000000, 0, None
    )

    # 等待发送端连接
    win32pipe.ConnectNamedPipe(pipe_receive, None)
    # 等待发送端连接
    win32pipe.ConnectNamedPipe(pipe_send, None)
    print("Sender connected.")
    print("receive connected.")

    return pipe_receive, pipe_send

def process_images(pipe_receive, pipe_send):
    image_count = 0
    batch_size = 5
    images = []

    while True:
        for i in range(5):
            start_time = time.time()  # 记录开始时间
            img_data = receive_data(pipe_receive)

            if img_data:
                image = Image.open(io.BytesIO(img_data))
                image = image.convert("L")  # 示例处理：转换为灰度图
                buf = io.BytesIO()
                image.save(buf, format='JPEG')
                buf.seek(0)  # 重置buffer位置到开始
                processed_data = buf.getvalue()

                images.append(buf)  # 存储 BytesIO 对象而不是 Image 对象

                if len(images) >= batch_size:
                    # 发送最后一个处理后的图像
                    send_image_back_to_qt(pipe_send, images[-1].getvalue())
                    images = []  # 清空列表以开始新的批处理
            time.sleep(0.01)  # 添加适当的延迟,降低CPU使用率
            print("Image received and saved.")
            end_time = time.time()  # 记录结束时间
            duration_ms = (end_time - start_time) * 1000  # 转换为毫秒
            print(f"Image {i + 1} received and displayed in {duration_ms:.2f} ms.")  # 打印毫秒级的时间
            image_count += 1  # 图片计数器增加
            print(f"Image {image_count} received and displayed.")

def main():
    pipe_receive_name = r'\\.\pipe\pipe_receive'
    pipe_send_name = r'\\.\pipe\pipe_send'
    pipe_receive, pipe_send = create_pipes(pipe_receive_name, pipe_send_name)
    process_images(pipe_receive, pipe_send)

if __name__ == '__main__':
    main()