# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 15:04
# @Author  : TG
# @File    : main.py
# @Software: PyCharm

import socket
import sys
import numpy as np
import cv2
import root_dir
import time
import os
from root_dir import ROOT_DIR
from utils import PreSocket, receive_sock, parse_protocol, ack_sock, done_sock, DualSock, simple_sock, test_sock
import logging
from utils import threshold_segmentation, largest_connected_component, draw_tomato_edge, bitwise_and_rgb_with_binary, extract_s_l, get_tomato_dimensions, get_defect_info
from collections import deque
import time


def process_cmd(cmd: str, img: any, connected_sock: socket.socket) -> tuple:
    """
    处理指令

    :param cmd: 指令类型
    :param data: 指令内容
    :param connected_sock: socket
    :param detector: 模型
    :return: 是否处理成功
    """
    start_time = time.time()
    if cmd == 'IM':
        # image = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = img
        threshold_s_l = 180
        # threshold_r_b = 15

        s_l = extract_s_l(img)
        # otsu_thresholded = ImageProcessor.otsu_threshold(s_l)
        # img_fore = ImageProcessor.bitwise_and_rgb_with_binary(img, otsu_thresholded)
        # img_fore_defect = ImageProcessor.extract_g_r(img_fore)
        # img_fore_defect = ImageProcessor.threshold_segmentation(img_fore_defect, threshold_r_b)

        # cv2.imshow('img_fore_defect', img_fore_defect)

        thresholded_s_l = threshold_segmentation(s_l, threshold_s_l)
        new_bin_img = largest_connected_component(thresholded_s_l)
        # zhongggggg = cv2.bitwise_or(new_bin_img, cv2.imread('defect_mask.bmp', cv2.IMREAD_GRAYSCALE))

        # cv2.imshow('zhongggggg', zhongggggg)

        # new_otsu_bin_img = ImageProcessor.largest_connected_component(otsu_thresholded)
        # filled_img, defect = ImageProcessor.fill_holes(new_bin_img)
        # defect = ImageProcessor.bitwise_and_rgb_with_binary(cv2.imread(img), defect)

        # cv2.imshow('defect', defect)

        edge, mask = draw_tomato_edge(img, new_bin_img)
        org_defect = bitwise_and_rgb_with_binary(edge, new_bin_img)
        # fore = ImageProcessor.bitwise_and_rgb_with_binary(cv2.imread(img), mask)
        # fore_g_r_t = ImageProcessor.threshold_segmentation(ImageProcessor.extract_g_r(fore), 20)
        # fore_g_r_t_ture = ImageProcessor.bitwise_and_rgb_with_binary(cv2.imread(img), fore_g_r_t)

        # cv2.imwrite('defect_big.bmp', fore_g_r_t_ture)
        
        # res = cv2.bitwise_or(new_bin_img, fore_g_r_t)
        # white = ImageProcessor.find_reflection(img)
        # cv2.imwrite('new_bin_img.bmp', new_bin_img)

        long_axis, short_axis = get_tomato_dimensions(mask)
        number_defects, total_pixels = get_defect_info(new_bin_img)
        rp = org_defect
        rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('rp1.bmp', rp)




    # elif cmd == 'TR':
    #     detector = WoodClass(w=4096, h=1200, n=8000, p1=0.8, debug_mode=False)
    #     model_name = None
    #     if "$" in data:
    #         data, model_name = data.split("$", 1)
    #         model_name = model_name + ".p"
    #     settings.data_path = data
    #     settings.model_path = ROOT_DIR / 'models' / detector.fit_pictures(data_path=settings.data_path, file_name=model_name)
    #     response = simple_sock(connected_sock, cmd_type=cmd, result=result)
    # elif cmd == 'MD':
    #     settings.model_path = data
    #     detector.load(path=settings.model_path)
    #     response = simple_sock(connected_sock, cmd_type=cmd, result=result)
    # elif cmd == 'KM':
    #     x_data, y_data, labels, img_names = detector.get_luminance_data(data, plot_2d=True)
    #     result = detector.data_adjustments(x_data, y_data, labels, img_names)
    #     result = ','.join([str(x) for x in result])
    #     response = simple_sock(connected_sock, cmd_type=cmd, result=result)


    else:
        logging.error(f'错误指令，指令为{cmd}')
        response = False

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'处理时间：{elapsed_time}秒')
    return long_axis, short_axis, number_defects, total_pixels, rp




def main(is_debug=False):
    file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'report.log'))
    file_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if is_debug else logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler],
                        level=logging.DEBUG)

    dual_sock = DualSock(connect_ip='127.0.0.1')

    while not dual_sock.status:
        dual_sock.reconnect()

    while True:
        long_axis_list = []
        short_axis_list = []
        defect_num_sum = 0
        total_defect_area_sum = 0
        rp = None

        for i in range(5):

            start_time = time.time()

            pack, next_pack = receive_sock(dual_sock)
            if pack == b"":
                time.sleep(2)
                dual_sock.reconnect()
                continue

            cmd, img = parse_protocol(pack)
            print(cmd)
            print(img.shape)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'接收时间：{elapsed_time}秒')

            long_axis, short_axis, number_defects, total_pixels, rp = process_cmd(cmd=cmd, img=img, connected_sock=dual_sock)
            # print(long_axis, short_axis, number_defects, type(total_pixels), rp.shape)

            if i <= 2:
                long_axis_list.append(long_axis)
                short_axis_list.append(short_axis)
            if i == 1:
                rp_result = rp

            defect_num_sum += number_defects
            total_defect_area_sum += total_pixels


            long_axis = round(sum(long_axis_list) / 3)
            short_axis = round(sum(short_axis_list) / 3)
        # print(type(long_axis), type(short_axis), type(defect_num_sum), type(total_defect_area_sum), type(rp_result))
        response = test_sock(dual_sock, cmd_type=cmd, long_axis=long_axis, short_axis=short_axis,
                            defect_num=defect_num_sum, total_defect_area=total_defect_area_sum, rp=rp_result)
        print(long_axis, short_axis, defect_num_sum, total_defect_area_sum, rp_result.shape)

    # while True:
    #     result_buffer = []
    #     for _ in range(5):
    #         pack, next_pack = receive_sock(dual_sock)  # 接收数据，如果没有数据则阻塞，如果返回的是空字符串则表示出现错误
    #         if pack == b"":  # 无数据表示出现错误
    #             time.sleep(5)
    #             dual_sock.reconnect()
    #             break
    #
    #         cmd, data = parse_protocol(pack)
    #         print(cmd)
    #         print(data)
    #
    #         result = process_cmd(cmd=cmd, data=data, connected_sock=dual_sock, detector=detector, settings=settings)
    #         result_buffer.append(result)  # 将处理结果添加到缓冲区
    #
    #     # 在这里进行对5次结果的处理，可以进行合并、比较等操作
    #     final_result = combine_results(result_buffer)
    #
    #     # 发送最终结果
    #     response = simple_sock(dual_sock, cmd_type=cmd, result=final_result)
    #     print(final_result)
    #     result_buffer = []



if __name__ == '__main__':
    # 2个端口
    # 接受端口21122
    # 发送端口21123
    # 接收到图片 n_rows * n_bands * n_cols, float32
    # 发送图片 n_rows * n_cols, uint8
    main(is_debug=False)
    # test(r"D:\build-tobacco-Desktop_Qt_5_9_0_MSVC2015_64bit-Release\calibrated15.raw")
    # main()
    # debug_main()
    # test_run(all_data_dir=r'D:\数据')
    # with open(r'D:\数据\虫子\valid2.raw', 'rb') as f:
    #     data = np.frombuffer(f.read(), dtype=np.float32).reshape(600, 29, 1024).transpose(0, 2, 1)
    # plt.matshow(data[:, :, 10])
    # plt.show()
    # detector = SpecDetector('model_spec/model_29.p')
    # result = detector.predict(data)
    #
    # plt.matshow(result)
    # plt.show()
    # result = result.reshape((600, 1024))