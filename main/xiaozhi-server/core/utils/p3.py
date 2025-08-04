import struct

def decode_opus_from_file(input_file):
    """
    从p3文件中解码 Opus 数据，并返回一个 Opus 数据包的列表以及总时长。
    """
    opus_datas = []

    with open(input_file, 'rb') as f:
        while True:
            # 读取头部（4字节）：[1字节类型，1字节保留，2字节长度]
            header = f.read(4)
            if not header:
                break

            # 解包头部信息
            _, _, data_len = struct.unpack('>BBH', header)

            # 根据头部指定的长度读取 Opus 数据
            opus_data = f.read(data_len)
            if len(opus_data) != data_len:
                raise ValueError(f"Data length({len(opus_data)}) mismatch({data_len}) in the file.")

            opus_datas.append(opus_data)

    return opus_datas

def decode_opus_from_bytes(input_bytes):
    """
    从p3二进制数据中解码 Opus 数据，并返回一个 Opus 数据包的列表以及总时长。
    """
    import io
    opus_datas = []

    f = io.BytesIO(input_bytes)
    while True:
        header = f.read(4)
        if not header:
            break
        _, _, data_len = struct.unpack('>BBH', header)
        opus_data = f.read(data_len)
        if len(opus_data) != data_len:
            raise ValueError(f"Data length({len(opus_data)}) mismatch({data_len}) in the bytes.")
        opus_datas.append(opus_data)

    return opus_datas