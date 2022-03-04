def threshold_equalization_old(self,
        thl_step=1,
        noise_limit=0,
        thl_offset=0,
        n_evals=3,
        use_gui=False
    ):

    thl_range = self.get_thl_range(thl_step=thl_step)

    print('== Threshold equalization ==')
    if use_gui:
        yield {'stage': 'Init'}

    # Set PC Mode in OMR in order to read kVp values
    omr = self.set_pc_mode()
    print('OMR set to:', omr)

    # Linear dependence: start and end points are sufficient
    pixel_dacs = ['00', '3f']

    # Return status to GUI
    if use_gui:
        yield {'stage': 'THL_pre_start'}

    counts_dict_gen = self.get_thl_level(
        thl_range, pixel_dacs, n_evals=n_evals, use_gui=use_gui)
    
    if use_gui:
        for res in counts_dict_gen:
            if 'status' in res.keys():
                yield {'stage': 'THL_pre', 'status': np.round(res['status'], 4)}
            elif 'DAC' in res.keys():
                yield {'stage': 'THL_pre_loop_start', 'status': res['DAC']}
            else:
                counts_dict = res['countsDict']
    else:
        counts_dict = deque(counts_dict_gen, maxlen=1).pop()

    gauss_dict, noise_thl = support.get_noise_level(
        counts_dict, thl_range, pixel_dacs, noise_limit)

    # Transform values to indices and get mean_dict
    mean_dict, noise_thl = support.val_to_idx(
        pixel_dacs, gauss_dict, noise_thl,
        self.dpx.thl_edges_low,
        self.dpx.thl_edges_high,
        self.dpx.thl_fit_params)

    # Calculate slope, offset and mean
    slope = (noise_thl['00'] - noise_thl['3f']) / 63.
    offset = noise_thl['00']

    # Fill missing values
    offset[np.isnan(offset)] = np.nanmedian(offset)
    slope[np.isnan(slope)] = np.nanmedian(slope)

    # mean = 0.5 * (mean_dict['00'] + mean_dict['3f'])
    thl_mean = mean_dict['3f'] + 3 * np.std(gauss_dict['3f'])

    print(offset)
    print(slope)
    for pixel in range(256):
        plt.plot([0, 1], [noise_thl['00'][pixel], noise_thl['3f'][pixel]], marker='x')
    plt.axhline(y=thl_mean, ls='--')
    plt.show()

    # Get adjustment value for each pixel
    adjust = np.asarray((offset - thl_mean) / slope + 0.5, dtype=int)

    # Consider extreme values
    adjust[np.isnan(adjust)] = 0
    adjust[adjust > 63] = 63
    adjust[adjust < 0] = 0

    # Set new pixel_dac values, convert to hex
    pixel_dac_new = ''.join(['%02x' % entry for entry in adjust.flatten()])
    print('New pixel dac', pixel_dac_new)

    # Repeat procedure to get noise levels
    counts_dict_new_gen = self.get_thl_level(
        thl_range, [pixel_dac_new], n_evals=n_evals, use_gui=use_gui)
    if use_gui:
        yield {'stage': 'THL_start'}
        for res in counts_dict_new_gen:
            if 'status' in res.keys():
                yield {'stage': 'THL', 'status': np.round(res['status'], 4)}
            elif 'DAC' in res.keys():
                yield {'stage': 'THL_loop_start', 'status': res['DAC']}
            else:
                counts_dict_new = res['countsDict']
    else:
        counts_dict_new = deque(counts_dict_new_gen, maxlen=1).pop()

    gauss_dict_new, noise_thl_new = support.get_noise_level(
        counts_dict_new, thl_range, [pixel_dac_new], noise_limit)

    # Transform values to indices
    _, noise_thl_new = support.val_to_idx(
        [pixel_dac_new], gauss_dict_new, noise_thl_new,
        self.dpx.thl_edges_low,
        self.dpx.thl_edges_high,
        self.dpx.thl_fit_params)

    plt.hist(noise_thl['00'], bins=30)
    plt.hist(noise_thl['3f'], bins=30)
    plt.hist(noise_thl_new[pixel_dac_new], bins=30)
    plt.show()

    # Plot the results of the equalization
    if use_gui:
        yield {'stage': 'conf_bits'}

    # Create conf_bits
    conf_mask = np.zeros(256).astype(str)
    conf_mask.fill('00')

    # Check for noisy pixels after equalization. If there are still any left,
    # reduce THL even further. If a pixel is really noisy, it shouldn't change
    # its state even when THL is lowered. Therefore, if pixels don't change their
    # behavior after 5 decrements of THL, switch them off
    if use_gui:
        yield {'stage': 'noise'}

    thl_new = int(np.median(gauss_dict_new[pixel_dac_new]))

    if self.dpx.thl_edges is not None:
        print('Getting rid of noisy pixels...')
        self.write_periphery(
            self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl_new)))

        pc_noisy_last = []
        noisy_count = 0
        while True:
            pc_data = np.zeros(256)
            for _ in range(10):
                self.data_reset()
                # time.sleep(0.01)
                pc_data += np.asarray(self.read_pc())
            # plt.imshow(pc_data.reshape((16, 16)))
            # plt.show()

            # Noisy pixels
            pc_noisy = np.argwhere(pc_data > 0).flatten()
            print('THL: %d' % thl_new)
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
            if len(pc_noisy) == len(noisy_common) and\
                len(pc_noisy) == len(pc_noisy_last):
                noisy_count += 1
            pc_noisy_last = np.array(pc_noisy, copy=True)

            # If noisy pixels don't change for 5 succeeding steps,
            # interrupt
            if noisy_count >= 10: # or len(pc_noisy) > 0:
                break

            # Reduce THL by 1
            thl_new = self.dpx.thl_edges[list(self.dpx.thl_edges).index(thl_new) - 1]
            self.write_periphery(
                self.dpx.periphery_dacs[:-4] + ('%04x' % int(thl_new)))

        # Subtract additional offset to THL
        # thl_new = self.dpx.thl_edges[list(self.dpx.thl_edges).index(thl_new) - thl_offset]

        # Switch off noisy pixels
        conf_mask[pc_data > 10] = '%02x' % (0b1 << 2)
    else:
        thl_new = int(np.mean(gauss_dict_new[pixel_dac_new]) - thl_offset)
        conf_mask[abs(noise_thl_new[pixel_dac_new] - thl_mean) > 10] = '%02x' % (0b1 << 2)
        print('Bad pixels:', np.argwhere(
            (abs(noise_thl_new[pixel_dac_new] - thl_mean) > 10)).flatten())

    # Transform into string
    conf_mask = ''.join(conf_mask.flatten())

    print()
    print('Summary:')
    print('pixel_DACs:', pixel_dac_new)
    print('confMask:', conf_mask)
    print('THL:', '%04x' % int(thl_new))

    # Restore OMR values
    self.write_omr(self.dpx.omr)

    if use_gui:
        yield {'stage': 'finished',
                'pixel_DAC': pixel_dac_new,
                'THL': '%04x' % int(thl_new),
                'confMask': conf_mask}
    else:
        yield pixel_dac_new, '%04x' % int(thl_new), conf_mask
