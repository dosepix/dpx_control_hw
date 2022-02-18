#!/usr/bin/env python
import dpx_control_hw as dch
import time

PORT = '/dev/ttyACM0'
def main():
    dpx = dch.Dosepix(port_name=PORT)
    ser = dpx.get_serial()

    # Send
    idx = 0

    '''
    while True:
        # ser.write(b'%d' % idx)
        ser.write(b'#02')
        # print(ser.readline())
        time.sleep(1)
        ser.write(b'#01')
        # print(ser.readline())
        time.sleep(1)

        ser.write(b'#07')
        time.sleep(1)

        idx += 1
    '''

if __name__ == '__main__':
    main()
