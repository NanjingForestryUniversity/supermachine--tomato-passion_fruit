
import socket
import time

import numpy as np
import logging
import os
import sys
import cv2 as cv

from classfier import Astragalin
from utils import DualSock, try_connect, receive_sock, parse_protocol, ack_sock, done_sock
from root_dir import ROOT_DIR
from model import resnet34
import torch
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
from PIL import Image



def process_cmd(cmd: str, data: any, connected_sock: socket.socket) -> tuple:
    '''
    处理指令
    :param cmd: 指令类型
    :param data: 指令内容
    :param connected_sock: socket
    :param detector: 模型
    :return: 是否处理成功
    '''
    result = ''
    if cmd == 'IM':
        data = np.frombuffer(data, dtype=np.uint8)
        data = cv.imdecode(data, cv.IMREAD_COLOR)

        # 显示图片
        cv.imshow('image', data)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        # data_transform = transforms.Compose(
        #     [transforms.Resize(256),
        #      transforms.CenterCrop(224),
        #      transforms.ToTensor(),
        #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #
        # # load image
        # # img_path = r"D:\project\deep-learning-for-image-processing-master\data_set\test_image\1.jpg"
        # # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        # # img = Image.open(img_path)
        # # plt.imshow(img)
        # # [N, C, H, W]
        # img = data_transform(data)
        # # expand batch dimension
        # img = torch.unsqueeze(img, dim=0)
        #
        # # read class_indict
        # # json_path = './class_indices.json'
        # # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        # #
        # # with open(json_path, "r") as f:
        # #     class_indict = json.load(f)
        #
        # # create model
        # model = resnet34(num_classes=4).to(device)
        #
        # # load model weights
        # weights_path = r"D:\project\deep-learning-for-image-processing-master\pytorch_classification\Test5_resnet\resNet34.pth"
        # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        # model.load_state_dict(torch.load(weights_path, map_location=device))
        #
        # # prediction
        # model.eval()
        # with torch.no_grad():
        #     # predict class
        #     output = torch.squeeze(model(img.to(device))).cpu()
        #     predict = torch.softmax(output, dim=0)
        #     predict_cla = torch.argmax(predict).numpy()
        #     result = predict_cla
            # print(predict_cla)
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()


        # result = detector.predict(data)
        # # 取出result中的字典中的centers和categories
        # centers = result['centers']
        # categories = result['categories']
        # # 将centers和categories转换为字符串，每一位之间用,隔开，centers是list,每个元素为np.array，categories是1维数组
        # # centers_str = '|'.join([str(point[0][0]) + ',' + str(point[0][1]) for point in centers])
        # # categories_str = ','.join([str(i) for i in categories])
        # # # 将centers和categories的字符串拼接起来，中间用;隔开
        # # result = centers_str + ';' + categories_str
        # 给result直接赋值，用于测试
        # result = 'HELLO WORLD'
        # response = done_sock(connected_sock, cmd_type=cmd, result=result)
        # print(result)
    else:
        logging.error(f'错误指令，指令为{cmd}')
        # response = False
    return result

def bytes_to_img(data):
    data1 = Image.frombytes('RGB', (1200, 4096), data, 'raw')





# def main(is_debug=False):
#     file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
#     file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
#     logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
#                         handlers=[file_handler, console_handler], level=logging.DEBUG)
#     dual_sock = DualSock(connect_ip='127.0.0.1')
#
#     while not dual_sock.status:
#         logging.error('连接被断开，正在重连')
#         dual_sock.reconnect()
#     detector = Astragalin(ROOT_DIR / 'models' / 'astragalin.p')
#     # _ = detector.predict(np.ones((4096, 1200, 10), dtype=np.float32))
#     while True:
#         pack, next_pack = receive_sock(dual_sock) # 接收数据，如果没有数据则阻塞，如果返回的是空字符串则表示出现错误
#         if pack == b"":  # 无数据表示出现错误
#             time.sleep(5)
#             dual_sock.reconnect()
#             continue
#
#         cmd, data = parse_protocol(pack)
#         print(cmd)
#         # print(data)
#
#         process_cmd(cmd=cmd, data=data, connected_sock=dual_sock, detector=detector)


def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler], level=logging.DEBUG)
    dual_sock = DualSock(connect_ip='127.0.0.1')

    while not dual_sock.status:
        logging.error('连接被断开，正在重连')
        dual_sock.reconnect()



    # detector = Astragalin(ROOT_DIR / 'models' / 'resNet34.pth')
    result_buffer = []  # 存储处理结果的缓冲区

    while True:
        for _ in range(5):
            pack, next_pack = receive_sock(dual_sock)  # 接收数据，如果没有数据则阻塞，如果返回的是空字符串则表示出现错误
            if pack == b"":  # 无数据表示出现错误
                time.sleep(5)
                dual_sock.reconnect()
                break

            cmd, data = parse_protocol(pack)
            print(cmd)
            # print(data)

            result = process_cmd(cmd=cmd, data=data, connected_sock=dual_sock)
            result_buffer.append(result)  # 将处理结果添加到缓冲区

        # 在这里进行对5次结果的处理，可以进行合并、比较等操作
        final_result = combine_results(result_buffer)

        # 发送最终结果
        response = done_sock(dual_sock, cmd_type=cmd, result=final_result)
        print(final_result)
        result_buffer = []



def combine_results(results):
    # 在这里实现对5次结果的合并/比较等操作，根据实际需求进行修改
    # 这里只是简单地将结果拼接成一个字符串，你可能需要根据实际情况进行更复杂的处理
    return ';'.join(results)


if __name__ == '__main__':
    main()