from . import communicate
import numpy as np
import time

class DPXFunctions():
    def __init__(self, dpx, comm):
        self.dpx = dpx
        self.comm = comm

    # === HARDWARE ===
    def enable_vdd(self):
        self.comm.send_cmd('EN_VDD')

    def disable_vdd(self):
        self.comm.send_cmd('DISAB_VDD')

    # === RESET ===
    def global_reset(self):
        self.comm.send_cmd('GLOBAL_RESET')

    def data_reset(self):
        self.comm.send_cmd('DATA_RESET')

    # === OMR ===
    def read_omr(self):
        self.comm.send_cmd('READ_OMR')
        return self.comm.get_data()

    def write_omr(self, data):
        self.comm.send_cmd('WRITE_OMR')
        self.comm.send_data_binary(data)

    def set_pc_mode(self, omr):
        omr_code = '%04x' % (
            (int(omr, 16) & ~((0b11) << 22)) | (0b10 << 22))
        self.write_omr(omr_code)
        return omr_code

    def set_dosi_mode(self, omr):
        omr_code = int(omr, 16) & ~((0b11) << 22)
        self.write_omr(omr_code)
        return omr_code

    # === PERIPHERY ====
    def read_periphery(self):
        self.comm.send_cmd('READ_PERIPHERY')
        return self.comm.get_data()

    def write_periphery(self, data):
        self.comm.send_cmd('WRITE_PERIPHERY')
        self.comm.send_data_binary(data)

    # === PIXEL DAC ===
    def read_pixel_dacs(self):
        self.comm.send_cmd('READ_PIXEL_DAC')
        return self.comm.get_data()

    def write_pixel_dacs(self, data):
        self.comm.send_cmd('WRITE_PIXEL_DAC')
        self.comm.send_data_binary(data)

    # === DATA ===
    def read_tot(self):
        self.comm.send_cmd('READ_TOT')
        return self.comm.get_data()

    # === FUNCTIONS ===
    def measure_tot(self):
        # Activate dosi mode
        self.dpx.omr = self.set_dosi_mode(self.dpx.omr)

        # Data reset
        self.data_reset()

        print('Starting ToT Measurement!')
        print('=========================')
        try:
            start_time = time.time()
            while True:
                data = self.read_tot()

        except (KeyboardInterrupt, SystemExit):
            raise

    '''
    def threshold_equalization(self, 
            omr,
            thl_edges=None,
            thl_step=1,
            noise_limit=3,
            use_gui=False
        ):

        # Get THL range
        thl_low, thl_high = 4000, 6000
        if len(thl_edges) == 0 or thl_edges is None:
            thl_range = np.arange(thl_low, thl_high)
        else:
            thl_range = np.asarray(thl_edges)
            thl_range = np.around(
                    thl_range[np.logical_and(thl_range >= thl_low, thl_range <= thl_high)]
                )

        print('== Threshold equalization ==')
        if use_gui:
            yield {'stage': 'Init'}

        # Set PC Mode in OMR in order to read kVp values
        self.set_PC_mode(OMR)

        # Linear dependence:
        # Start and end points are sufficient
        pixel_dacs = ['00', '3f']

        # Return status to GUI
        if use_gui:
            yield {'stage': 'THL_pre_start'}
            counts_dict_gen = self.get_thl_level(
                slot, THLRange, pixel_DACs, reps, intPlot=False, use_gui=True)
            for res in counts_dict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL_pre', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_pre_loop_start', 'status': res['DAC']}
                else:
                    countsDict = res['countsDict']
            # countsDict = self.getTHLLevel_gui(slot, THLRange, pixel_DACs, reps)
        else:
            countsDict = self.get_thl_level(
                slot, THLRange, pixel_DACs, reps, intPlot, use_gui=False)
        gaussDict, noiseTHL = self.getNoiseLevel(
            countsDict, THLRange, pixel_DACs, noiseLimit)

        # Transform values to indices and get mean_dict
        mean_dict, noiseTHL = self.valToIdx(
            slot, pixel_DACs, THLRange, gaussDict, noiseTHL)

        if len(pixel_DACs) > 2:
            def slopeFit(x, m, t):
                return m * x + t
            slope = np.zeros((16, 16))
            offset = np.zeros((16, 16))

        else:
            slope = (noiseTHL['00'] - noiseTHL['3f']) / 63.
            offset = noiseTHL['00']

        x = [int(key, 16) for key in pixel_DACs]
        if len(pixel_DACs) > 2:
            # Store fit functions in list
            polyCoeffList = []

        for pixelX in range(16):
            for pixelY in range(16):
                y = []
                for pixel_DAC in pixel_DACs:
                    y.append(noiseTHL[pixel_DAC][pixelX, pixelY])

        mean = 0.5 * (mean_dict['00'] + mean_dict['3f'])
        # print mean_dict['00'], mean_dict['3f'], mean

        # Get adjustment value for each pixel
        adjust = np.asarray((offset - mean) / slope + 0.5)

        # Consider extreme values
        adjust[np.isnan(adjust)] = 0
        adjust[adjust > 63] = 63
        adjust[adjust < 0] = 0

        # Convert to integer
        adjust = adjust.astype(dtype=int)

        # Set new pixel_DAC values
        pixel_DACNew = ''.join(['%02x' % entry for entry in adjust.flatten()])

        # Repeat procedure to get noise levels
        if use_gui:
            yield {'stage': 'THL_start'}
            countsDict_gen = self.getTHLLevel(
                slot, THLRange, pixel_DACNew, reps, intPlot=False, use_gui=True)
            for res in countsDict_gen:
                if 'status' in res.keys():
                    yield {'stage': 'THL', 'status': np.round(res['status'], 4)}
                elif 'DAC' in res.keys():
                    yield {'stage': 'THL_loop_start', 'status': res['DAC']}
                else:
                    countsDictNew = res['countsDict']
        else:
            countsDictNew = self.getTHLLevel(
                slot, THLRange, pixel_DACNew, reps, intPlot)
        gaussDictNew, noiseTHLNew = self.getNoiseLevel(
            countsDictNew, THLRange, pixel_DACNew, noiseLimit)

        # Transform values to indices
        mean_dictNew, noiseTHLNew = self.valToIdx(
            slot, [pixel_DACNew], THLRange, gaussDictNew, noiseTHLNew)

        # Plot the results of the equalization
        if resPlot and not use_gui:
            bins = np.linspace(min(gaussDict['3f']), max(gaussDict['00']), 100)

            for pixel_DAC in ['00', '3f']:
                plt.hist(
                    gaussDict[pixel_DAC],
                    bins=bins,
                    label='%s' %
                    pixel_DAC,
                    alpha=0.5)

            plt.hist(
                gaussDictNew[pixel_DACNew],
                bins=bins,
                label='After equalization',
                alpha=0.5)

            plt.legend()

            plt.xlabel('THL')
            plt.ylabel('Counts')

            plt.show()

        if use_gui:
            yield {'stage': 'conf_bits'}

        # Create conf_bits
        confMask = np.zeros((16, 16)).astype(str)
        confMask.fill('00')

        # Check for noisy pixels after equalization. If there still are any left, reduce THL value even further
        # to reduce their amount. If a pixel is really noisy, it shouldn't change its state even when THL is lowered.
        # Therefore, if pixels don't change their behavior after 5 decrements
        # of THL, switch those pixels off.
        if use_gui:
            yield {'stage': 'noise'}

        THLNew = int(np.mean(gaussDictNew[pixel_DACNew]))

        if self.THL_edges[slot - 1] is not None:
            print('Getting rid of noisy pixels...')
            self.DPX_write_periphery_DAC_command(
                slot, self.peripherys[slot - 1] + ('%04x' % int(THLNew)))

            pc_noisy_last = []
            noisy_count = 0
            while True:
                pc_data = np.zeros((16, 16))
                self.DPX_data_reset_command(slot)
                for cnt in range(30):
                    pc_data += np.asarray(self.DPXReadToTDatakVpModeCommand(slot))
                    self.DPX_data_reset_command(slot)
                pc_sum = pc_data.flatten()

                # Noisy pixels
                pc_noisy = np.argwhere(pc_sum > 0).flatten()
                print('THL: %d' % THLNew)
                print('Noisy pixels index:')
                print(pc_noisy)
                # Compare with previous read-out
                noisy_common = sorted(list(set(pc_noisy_last) & set(pc_noisy)))
                print(noisy_common)
                print(len(pc_noisy), len(noisy_common))
                print(noisy_count)
                print()

                # If noisy pixels don't change, increase counter
                # if len(list(set(noisy_common) & set(pc_noisy))) > 0:
                if len(pc_noisy) == len(noisy_common) and len(
                        pc_noisy) == len(pc_noisy_last):
                    noisy_count += 1
                pc_noisy_last = np.array(pc_noisy, copy=True)

                # If noisy pixels don't change for 5 succeeding steps,
                # interrupt
                if noisy_count == 3 or not len(pc_noisy):
                    break
                else:
                    # Reduce THL by 1
                    THLNew = self.THL_edges[slot -
                                           1][list(self.THL_edges[slot -
                                                                 1]).index(THLNew) -
                                              1]
                    self.DPX_write_periphery_DAC_command(
                        slot, self.peripherys[slot - 1] + ('%04x' % int(THLNew)))

            # Subtract additional offset to THL
            THLNew = self.THL_edges[slot -
                                   1][list(self.THL_edges[slot -
                                                         1]).index(THLNew) -
                                      THL_offset]

            # Switch off noisy pixels
            confMask[(pc_sum > 10).reshape((16, 16))] = '%02x' % (0b1 << 2)
        else:
            THLNew = int(np.mean(gaussDictNew[pixel_DACNew]) - THL_offset)
            confMask[abs(noiseTHLNew[pixel_DACNew] - mean)
                     > 10] = '%02x' % (0b1 << 2)

        # Transform into string
        confMask = ''.join(confMask.flatten())

        print()
        print('Summary:')
        print('pixel_DACs:', pixel_DACNew)
        print('confMask:', confMask)
        print('Bad pixels:', np.argwhere(
            (abs(noiseTHLNew[pixel_DACNew] - mean) > 10)))
        print('THL:', '%04x' % int(THLNew))

        # Restore OMR values
        self.DPX_write_OMR_command(slot, self.OMR[slot - 1])

        if use_gui:
            return {'stage': 'finished',
                    'pixel_DAC': pixel_DACNew,
                    'THL': '%04x' % int(THLNew),
                    'confMask': confMask}
        else:
            return pixel_DACNew, '%04x' % int(THLNew), confMask

    def getTHLLevel(
            self,
            thl_range,
            pixel_dacs=['00', '3f'],
            use_gui=False
        ):
        counts_dict = {}

        # Loop over pixel_dac values
        for pixel_dac in pixel_dacs:
            counts_dict[pixel_dac] = {}
            print('Set pixel DACs to %s' % pixel_dac)

            # Set pixel DAC values
            pixel_code = pixel_dac * 256
            self.write_pixel_dacs(pixel_code)

            # Dummy readout
            for j in range(3):
                self.DPXReadToTDatakVpModeCommand()
                # time.sleep(0.2)

            # Noise measurement
            # Loop over THL values
            print('Loop over THLs')

            # Fast loop
            counts_list = []
            thl_range_fast = thl_range[::10]
            for cnt, thl in enumerate(thl_range_fast):
                self.write_periphery(
                    self.peripherys + ('%04x' % int(thl))
                )
                self.DPX_data_reset_command()
                time.sleep(0.001)

                # Read ToT values into matrix
                countsList.append(
                    self.DPXReadToTDatakVpModeCommand(slot).flatten())

            countsList = np.asarray(countsList).T
            thl_range_fast = [thl_range_fast[item[0][0]] if np.any(item) else np.nan for item in [
                np.argwhere(counts > 3) for counts in countsList]]

            # Precise loop
            if use_gui:
                yield {'DAC': pixel_DAC}

            THLRangeSlow = np.around(THLRange[np.logical_and(THLRange >= (
                np.nanmin(THLRangeFast) - 10), THLRange <= np.nanmax(THLRangeFast))])

            NTHL = len(THLRangeSlow)
            # Do not use tqdm with GUI
            if use_gui:
                loop_range = THLRangeSlow
            else:
                loop_range = tqdm(THLRangeSlow)
            for cnt, THL in enumerate(loop_range):
                # Repeat multiple times since data is noisy
                counts = np.zeros((16, 16))
                for lp in range(reps):
                    self.DPX_write_periphery_DAC_command(
                        slot, self.peripherys[slot - 1] + ('%04x' % int(THL)))
                    self.DPX_data_reset_command(slot)
                    time.sleep(0.001)

                    # Read ToT values into matrix
                    counts += self.DPXReadToTDatakVpModeCommand(slot)

                counts /= float(reps)
                countsDict[pixel_DAC][int(THL)] = counts

                # Return status as generator when using GUI
                if use_gui:
                    yield {'status': float(cnt) / len(loop_range)}
            print()
        if use_gui:
            yield {'countsDict': countsDict}
        else:
            return countsDict
    '''
