CMD_OK = "#CMD_OK"
DATA_OK = "#DATA_OK"

class Communicate():
    def __init__(self, ser, debug=False):
        self.ser = ser
        self.debug = debug

    def send_cmd(self, command, write=True):
        command += (20 - len(command)) * '?'

        if self.debug:
            print(command)
        if write:
            self.ser.write(str.encode('#' + command))
        else:
            self.ser.write(str.encode('!' + command))

        res = self.get_response()
        if self.debug:
            print(res)
            print()
        return res == CMD_OK

    def send_data(self, data):
        self.ser.write(str.encode(data))

        res = self.get_response()
        if self.debug:
            print(res)
            print()
        return res == DATA_OK

    def send_data_binary(self, data):
        self.ser.write(bytes.fromhex(data))
        res = self.get_response()
        if self.debug:
            print(res)
            print()
        return res == DATA_OK

    def get_response(self):
        res = self.ser.read_until()
        # Remove newline
        try:
            return res[:-1].decode()
        except UnicodeDecodeError:
            return 'DECODE_ERROR'

    def get_data(self, size=None):
        if size is None:
            res = self.ser.read_until()[:-1]
            if res[-len(DATA_OK):].decode() == DATA_OK:
                return res[:-len(DATA_OK)]
        else:
            res_data = self.ser.read(size=size)
            if self.debug:
                print(res_data)
                print()
            res_end = self.ser.read_until()[:-1]
            if res_end.decode() == DATA_OK:
                return res_data
        return ''
