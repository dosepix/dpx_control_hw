from collections import namedtuple

omr_analog_out_sel_type = namedtuple(
    "OMRAnalogOutSel",
    "v_tha v_tpref_fine v_casc_preamp\
        v_fbk v_tpref_coarse v_gnd i_preamp\
        i_disc1 i_disc2 v_tpbufout v_tpbufin\
        i_krum i_dac_pixel v_bandgap v_casc_krum\
        temperature v_per_bias v_cascode_bias high_z")
omr_analog_out_sel = omr_analog_out_sel_type(
    v_tha=0b00001 << 12,
    v_tpref_fine=0b00010 << 12,
    v_casc_preamp=0b00011 << 12,
    v_fbk=0b00100 << 12,
    v_tpref_coarse=0b00101 << 12,
    v_gnd=0b00110 << 12,
    i_preamp=0b00111 << 12,
    i_disc1=0b01000 << 12,
    i_disc2=0b01001 << 12,
    v_tpbufout=0b01010 << 12,
    v_tpbufin=0b01011 << 12,
    i_krum=0b01100 << 12,
    i_dac_pixel=0b01101 << 12,
    v_bandgap=0b01110 << 12,
    v_casc_krum=0b01111 << 12,
    temperature=0b11011 << 12,
    v_per_bias=0b11100 << 12,
    v_cascode_bias=0b11101 << 12,
    high_z=0b11111 << 12)
