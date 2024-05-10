
import cv2

import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import logging
import os
import shutil
import time
import socket



def read_envi_ascii(file_name, save_xy=False, hdr_file_name=None):
    """
    Read envi ascii file. Use ENVI ROI Tool -> File -> output ROIs to ASCII...

    :param file_name: file name of ENVI ascii file
    :param hdr_file_name: hdr file name for a "BANDS" vector in the output
    :param save_xy: save the x, y position on the first two cols of the result vector
    :return: dict {class_name: vector, ...}
    """
    number_line_start_with = "; Number of ROIs: "
    roi_name_start_with, roi_npts_start_with = "; ROI name: ", "; ROI npts:"
    data_start_with, data_start_with2, data_start_with3 = ";    ID", ";   ID", ";     ID"
    class_num, class_names, class_nums, vectors = 0, [], [], []
    with open(file_name, 'r') as f:
        for line_text in f:
            if line_text.startswith(number_line_start_with):
                class_num = int(line_text[len(number_line_start_with):])
            elif line_text.startswith(roi_name_start_with):
                class_names.append(line_text[len(roi_name_start_with):-1])
            elif line_text.startswith(roi_npts_start_with):
                class_nums.append(int(line_text[len(roi_name_start_with):-1]))
            elif line_text.startswith(data_start_with) or line_text.startswith(data_start_with2) or line_text.startswith(data_start_with3):
                col_list = list(filter(None, line_text[1:].split(" ")))
                assert (len(class_names) == class_num) and (len(class_names) == len(class_nums))
                break
            elif line_text.startswith(";"):
                continue
        for vector_rows in class_nums:
            vector_str = ''
            for i in range(vector_rows):
                vector_str += f.readline()
            vector = np.fromstring(vector_str, dtype=float, sep=" ").reshape(-1, len(col_list))
            assert vector.shape[0] == vector_rows
            vector = vector[:, 3:] if not save_xy else vector[:, 1:]
            vectors.append(vector)
            f.readline()  # suppose to read a blank line
    if hdr_file_name is not None:
        import re
        with open(hdr_file_name, 'r') as f:
            hdr_info = f.read()
        bands = re.findall(r"wavelength = {[^{}]+}", hdr_info, flags=re.IGNORECASE | re.MULTILINE)
        bands_num = re.findall(r"bands\s*=\s*(\d+)", hdr_info, flags=re.I)
        if (len(bands) == 0) or len(bands_num) == 0:
            Warning("The given hdr file is invalid, can't find bands = ? or wavelength = {?}.")
        else:
            bands = re.findall(r'{[^{}]+}', bands[0], flags=re.MULTILINE)[0][3:-2]
            bands = bands.split(',\n')
            bands = np.asarray(bands, dtype=float)
            bands_num = int(bands_num[0])
            if bands_num == bands.shape[0]:
                bands = np.array(bands, dtype=float)
                vectors.append(bands)
                class_names.append("BANDS")
            else:
                Warning("The given hdr file is invalid, bands num is not equal to wavelength.")
    return dict(zip(class_names, vectors))


def ga_feature_extraction(data_x, data_y):
    '''
    使用遗传算法进行特征提取
    :param data_x: 特征
    :param data_y: 类别
    '''
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x, data_y, test_size=0.3)
    clf = DecisionTreeClassifier(random_state=3)
    selector = GeneticSelectionCV(clf, cv=30,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=10,
                                  n_population=500,
                                  crossover_proba=0.6,
                                  mutation_proba=0.3,
                                  n_generations=300,
                                  crossover_independent_proba=0.6,
                                  mutation_independent_proba=0.1,
                                  tournament_size=10,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(Xtrain, Ytrain)
    Xtrain_ga, Xtest_ga = Xtrain[:, selector.support_], Xtest[:, selector.support_]
    clf = clf.fit(Xtrain_ga, Ytrain)
    print(np.where(selector.support_ == True))
    y_pred = clf.predict(Xtest_ga)
    print(classification_report(Ytest, y_pred))
    print(confusion_matrix(Ytest, y_pred))


def read_raw(file_name, shape=None,  setect_bands=None, cut_shape=None):
    '''
    读取raw文件
    :param file_name: 文件名
    :param setect_bands: 选择的波段
    :return: 波段数据
    '''
    if shape is None:
        shape = (692, 272, 384)
    with open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape).transpose(0, 2, 1)
    if setect_bands is not None:
        data = data[:, :, setect_bands]
    if cut_shape is not None:
        data = data[: cut_shape[0], : cut_shape[1], :]
    return data


def save_raw(file_name, data):
    '''
    保存raw文件
    :param file_name: 文件名
    :param data: 数据
    '''
    data = data.transpose(0, 2, 1)
    # 将data转换为一维数组
    data = data.reshape(-1)
    with open(file_name, 'wb') as f:
        f.write(data.astype(np.float32).tobytes())


def read_rgb(file_name):
    '''
    读取rgb文件
    :param file_name: 文件名
    :return: rgb数据
    '''
    data = cv2.imread(file_name)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    #给一个颜色对应的字典，用于将rgb转换为类别，白色对应0，黄色对应1，青色对应2，红色对应3，绿色对应4，蓝色对应5
    color_dict = {(255, 255, 255): 0, (255, 255, 0): 1, (0, 255, 255): 2, (255, 0, 0): 3, (0, 255, 0): 4, (0, 0, 255): 5}
    # 保存图片的形状，用于将一维数组转换为三维数组
    shape = data.shape
    # 将rgb转换为类别
    data = data.reshape(-1, 3).tolist()
    # 将rgb转换为类别
    mapped_data = []

    for i, color in enumerate(data):
        mapped_value = color_dict.get(tuple(color))
        if mapped_value is None:
            print("No mapping found for color", color, "at index", i)
        else:
            mapped_data.append(mapped_value)
    # 将一维数组转换为三维数组
    data = np.array(mapped_data).reshape(shape[0], shape[1])
    return data


def read_data(raw_path, rgb_path, shape=None, setect_bands=None, blk_size=4, cut_shape=None, dp=False):
    '''
    读取数据
    :param raw_path: raw文件路径
    :param rgb_path: rgb文件路径
    :param setect_bands: 选择的波段
    :return: 波段数据，rgb数据
    '''
    if shape is None:
        shape = (692, 272, 384)
    with open(raw_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.float32).reshape(shape).transpose(0, 2, 1)
    if setect_bands is not None:
        raw = raw[:, :, setect_bands]
    color_dict = {(255, 255, 255): 0, (255, 255, 0): 1, (0, 255, 255): 2, (255, 0, 0): 3, (0, 255, 0): 4,
                  (0, 0, 255): 5}
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if cut_shape is not None:
        raw = raw[ :cut_shape[0], :cut_shape[1], :]
        rgb = rgb[ :cut_shape[0], :cut_shape[1], :]
    data_x = []
    data_y = []
    for i in range(0, rgb.shape[0], blk_size):
        for j in range(0, rgb.shape[1], blk_size):
            x = raw[i:i + blk_size, j:j + blk_size, :]
            y = rgb[i:i + blk_size, j:j + blk_size]
            # # 取y的第三行第三列的像素值，判断该像素值是否在color_dict中，如果在则将x和y添加到data_x和data_y中
            # y = tuple(y[2, 2, :])
            # if y in color_dict.keys():
            #     data_x.append(x)
            #     data_y.append(color_dict[y])
            # 取y的中心点像素值，判断该像素值是否在color_dict中，如果在则将x和y添加到data_x和data_y中
            y = tuple(y[blk_size//2, blk_size//2, :])
            if y in color_dict.keys():
                data_x.append(x)
                data_y.append(color_dict[y])
    data_x = np.array(data_x)
    data_y = np.array(data_y).astype(np.uint8)
    return data_x, data_y


def try_connect(connect_ip: str, port_number: int, is_repeat: bool = False, max_reconnect_times: int = 50) -> (
                bool, socket.socket):
    """
    尝试连接.

    :param is_repeat: 是否是重新连接
    :param max_reconnect_times:最大重连次数
    :return: (连接状态True为成功, Socket / None)
    """
    reconnect_time = 0
    while reconnect_time < max_reconnect_times:
        logging.warning(f'尝试{"重新" if is_repeat else ""}发起第{reconnect_time + 1}次连接...')
        try:
            connected_sock = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
            connected_sock.connect((connect_ip, port_number))
        except Exception as e:
            reconnect_time += 1
            logging.error(f'第{reconnect_time}次连接失败... 5秒后重新连接...\n {e}')
            time.sleep(5)
            continue
        logging.warning(f'{"重新" if is_repeat else ""}连接成功')
        return True, connected_sock
    return False, None


class PreSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_pack = b''
        self.settimeout(5)

    def receive(self, *args, **kwargs):
        if self.pre_pack == b'':
            return self.recv(*args, **kwargs)
        else:
            data_len = args[0]
            required, left = self.pre_pack[:data_len], self.pre_pack[data_len:]
            self.pre_pack = left
            return required

    def set_prepack(self, pre_pack: bytes):
        temp = self.pre_pack
        self.pre_pack = temp + pre_pack


class DualSock(PreSocket):
    def __init__(self, connect_ip='127.0.0.1', recv_port: int = 21122, send_port: int = 21123):
        super().__init__()
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        self.status = received_status and send_status

    def send(self, *args, **kwargs) -> int:
        return self.send_sock.send(*args, **kwargs)

    def receive(self, *args, **kwargs) -> bytes:
        return self.received_sock.receive(*args, **kwargs)

    def set_prepack(self, pre_pack: bytes):
        self.received_sock.set_prepack(pre_pack)

    def reconnect(self, connect_ip='127.0.0.1', recv_port:int = 21122, send_port: int = 21123):
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        return received_status and send_status


def receive_sock(recv_sock: PreSocket, pre_pack: bytes = b'', time_out: float = -1.0, time_out_single=5e20) -> (
bytes, bytes):
    """
    从指定的socket中读取数据.自动阻塞，如果返回的数据为空则说明连接出现问题，需要重新连接。

    :param recv_sock: 指定sock
    :param pre_pack: 上一包的粘包内容
    :param time_out: 每隔time_out至少要发来一次指令,否则认为出现问题进行重连，小于0则为一直等
    :param time_out_single: 单次指令超时时间，单位是秒
    :return: data, next_pack
    """
    recv_sock.set_prepack(pre_pack)
    # 开头校验
    time_start_recv = time.time()
    while True:
        if time_out > 0:
            if (time.time() - time_start_recv) > time_out:
                logging.error(f'指令接收超时')
                return b'', b''
        try:
            temp = recv_sock.receive(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return b'', b''
        except TimeoutError as e:
            # logging.error(f'超时了，错误代码: \n{e}')
            logging.info('运行中,等待指令..')
            continue
        except socket.timeout as e:
            logging.info('运行中,等待指令..')
            continue
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return b'', b''
        if temp == b'\xaa':
            break

    # 接收开头后，开始进行时间记录
    time_start_recv = time.time()

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文的长度不正确, 错误代码: \n{e}')
            return b'', b''
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}')
        return b'', b''

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}')
            return b'', b''
    data, next_pack = temp[:data_len], temp[data_len:]
    recv_sock.set_prepack(next_pack)
    next_pack = b''

    # 进行数据校验
    temp = b''
    while len(temp) < 3:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文校验失败, 错误代码: \n{e}')
            return b'', b''
    if temp == b'\xff\xff\xbb':
        return data, next_pack
    else:
        logging.error(f"接收了一个完美的只错了校验位的报文")
        return b'', b''


def parse_protocol(data: bytes) -> (str, any):
    '''
    指令转换
    :param data: 接收到的报文
    :return: 指令类型，指令内容
    '''
    try:
        assert len(data) > 4
    except AssertionError:
        logging.error('指令转换失败，长度不足5')
        return '', None
    cmd, data = data[:4], data[4:]
    cmd = cmd.decode('ascii').strip().upper()
    if cmd == 'IM':
        n_rows, n_cols, img = data[:2], data[2:4], data[4:]
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols: {n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape((n_rows, n_cols, -1))
        return cmd, img


def ack_sock(send_sock: socket.socket, cmd_type: str) -> bool:
    '''
    发送应答
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :return:是否发送成功
    '''
    msg = b'\xaa\x00\x00\x00\x05' + (' A' + cmd_type).upper().encode('ascii') + b'\xff\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送应答失败，错误类型：{e}')
        return False
    return True


def done_sock(send_sock: socket.socket, cmd_type: str, result = '') -> bool:
    '''
    发送任务完成指令
    :param send_sock: 指定sock
    :param cmd_type: 指令类型
    :param result: 数据
    :return: 是否发送成功
    '''
    cmd = cmd_type.strip().upper()
    if cmd_type == 'IM':
        result = result.encode()
        # 指令4位
        length = len(result) + 4
        length = length.to_bytes(4, byteorder='big')
        # msg = b'\xaa' + length + (' D' + cmd).upper().encode('ascii') + result + b'\xff\xff\xbb'
        msg = result
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True

def simple_sock(send_sock: socket.socket, cmd_type: str, result) -> bool:
    '''
    发送任务完成指令
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :param result:数据
    :return:是否发送成功
    '''
    cmd_type = cmd_type.strip().upper()
    if cmd_type == 'IM':
        if result == 0:
            msg = b'S'
        elif result == 1:
            msg = b'Z'
        elif result == 2:
            msg = b'Q'
    elif cmd_type == 'TR':
        msg = b'A'
    elif cmd_type == 'MD':
        msg = b'D'
    elif cmd_type == 'KM':
        msg = b'K'
        result = result.encode('ascii')
        result = b',' + result
        length = len(result)
        msg = msg + length.to_bytes(4, 'big') + result
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True

def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def create_file(file_name):
    """
    创建文件
    :param file_name: 文件名
    :return: None
    """
    if os.path.exists(file_name):
        print("文件存在：%s" % file_name)
        return False
        # os.remove(file_name)  # 删除已有文件
    if not os.path.exists(file_name):
        print("文件不存在，创建文件：%s" % file_name)
        open(file_name, 'a').close()
        return True


class Logger(object):
    def __init__(self, is_to_file=False, path=None):
        self.is_to_file = is_to_file
        if path is None:
            path = "Astragalins.log"
        self.path = path
        create_file(path)

    def log(self, content):
        if self.is_to_file:
            with open(self.path, "a") as f:
                print(time.strftime("[%Y-%m-%d_%H-%M-%S]:"), file=f)
                print(content, file=f)
        else:
            print(content)