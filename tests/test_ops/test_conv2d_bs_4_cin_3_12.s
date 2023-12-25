
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_9_3_3_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_hi_i32 s2, s13, 0x55555556                           // 000000001714: 9702FF0D 55555556
	s_mul_i32 s0, s15, 0xe7                                    // 00000000171C: 9600FF0F 000000E7
	s_lshr_b32 s3, s2, 31                                      // 000000001724: 85039F02
	s_ashr_i32 s1, s0, 31                                      // 000000001728: 86019F00
	s_add_i32 s10, s2, s3                                      // 00000000172C: 810A0302
	s_lshl_b64 s[2:3], s[0:1], 2                               // 000000001730: 84828200
	s_mul_i32 s0, s10, 3                                       // 000000001734: 9600830A
	s_mul_i32 s10, s10, 7                                      // 000000001738: 960A870A
	s_sub_i32 s12, s13, s0                                     // 00000000173C: 818C000D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s1, s6, s2                                       // 000000001744: 80010206
	s_addc_u32 s6, s7, s3                                      // 000000001748: 82060307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001750: BF8704D9
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_mul_i32 s10, s14, 45                                     // 000000001758: 960AAD0E
	s_add_u32 s1, s1, s2                                       // 00000000175C: 80010201
	s_addc_u32 s7, s6, s3                                      // 000000001760: 82070306
	s_ashr_i32 s13, s12, 31                                    // 000000001764: 860D9F0C
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001768: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s6, s1, s2                                       // 000000001770: 80060201
	s_addc_u32 s7, s7, s3                                      // 000000001774: 82070307
	s_ashr_i32 s11, s10, 31                                    // 000000001778: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 00000000177C: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s8, s8, s10                                      // 000000001784: 80080A08
	s_addc_u32 s9, s9, s11                                     // 000000001788: 82090B09
	s_load_b512 s[36:51], s[8:9], null                         // 00000000178C: F4100904 F8000000
	s_clause 0x2                                               // 000000001794: BF850002
	s_load_b128 s[16:19], s[6:7], null                         // 000000001798: F4080403 F8000000
	s_load_b32 s1, s[6:7], 0x10                                // 0000000017A0: F4000043 F8000010
	s_load_b128 s[20:23], s[6:7], 0x1c                         // 0000000017A8: F4080503 F800001C
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s16, s36, 0                                  // 0000000017B4: D6130000 02004810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017BC: BF870091
	v_fmac_f32_e64 v0, s17, s37                                // 0000000017C0: D52B0000 00004A11
	v_fmac_f32_e64 v0, s18, s38                                // 0000000017C8: D52B0000 00004C12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017D0: BF8700A1
	v_fmac_f32_e64 v0, s19, s39                                // 0000000017D4: D52B0000 00004E13
	s_load_b128 s[36:39], s[6:7], 0x38                         // 0000000017DC: F4080903 F8000038
	v_fmac_f32_e64 v0, s1, s40                                 // 0000000017E4: D52B0000 00005001
	s_load_b32 s1, s[6:7], 0x2c                                // 0000000017EC: F4000043 F800002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F4: BF870091
	v_fmac_f32_e64 v0, s20, s41                                // 0000000017F8: D52B0000 00005214
	v_fmac_f32_e64 v0, s21, s42                                // 000000001800: D52B0000 00005415
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001808: BF8700A1
	v_fmac_f32_e64 v0, s22, s43                                // 00000000180C: D52B0000 00005616
	s_load_b128 s[40:43], s[6:7], 0x134                        // 000000001814: F4080A03 F8000134
	v_fmac_f32_e64 v0, s23, s44                                // 00000000181C: D52B0000 00005817
	s_load_b512 s[16:31], s[8:9], 0x40                         // 000000001824: F4100404 F8000040
	s_waitcnt lgkmcnt(0)                                       // 00000000182C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001830: BF8700A1
	v_fmac_f32_e64 v0, s1, s45                                 // 000000001834: D52B0000 00005A01
	s_load_b32 s1, s[6:7], 0x48                                // 00000000183C: F4000043 F8000048
	v_fmac_f32_e64 v0, s36, s46                                // 000000001844: D52B0000 00005C24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870091
	v_fmac_f32_e64 v0, s37, s47                                // 000000001850: D52B0000 00005E25
	v_fmac_f32_e64 v0, s38, s48                                // 000000001858: D52B0000 00006026
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001860: BF8700B1
	v_fmac_f32_e64 v0, s39, s49                                // 000000001864: D52B0000 00006227
	s_load_b128 s[36:39], s[6:7], 0x150                        // 00000000186C: F4080903 F8000150
	s_waitcnt lgkmcnt(0)                                       // 000000001874: BF89FC07
	v_fmac_f32_e64 v0, s1, s50                                 // 000000001878: D52B0000 00006401
	s_load_b32 s1, s[6:7], 0x144                               // 000000001880: F4000043 F8000144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870091
	v_fmac_f32_e64 v0, s40, s51                                // 00000000188C: D52B0000 00006628
	v_fmac_f32_e64 v0, s41, s16                                // 000000001894: D52B0000 00002029
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000189C: BF870091
	v_fmac_f32_e64 v0, s42, s17                                // 0000000018A0: D52B0000 0000222A
	v_fmac_f32_e64 v0, s43, s18                                // 0000000018A8: D52B0000 0000242B
	s_load_b128 s[40:43], s[6:7], 0x16c                        // 0000000018B0: F4080A03 F800016C
	s_waitcnt lgkmcnt(0)                                       // 0000000018B8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018BC: BF8700A1
	v_fmac_f32_e64 v0, s1, s19                                 // 0000000018C0: D52B0000 00002601
	s_load_b32 s1, s[6:7], 0x160                               // 0000000018C8: F4000043 F8000160
	v_fmac_f32_e64 v0, s36, s20                                // 0000000018D0: D52B0000 00002824
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D8: BF870091
	v_fmac_f32_e64 v0, s37, s21                                // 0000000018DC: D52B0000 00002A25
	v_fmac_f32_e64 v0, s38, s22                                // 0000000018E4: D52B0000 00002C26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018EC: BF8700C1
	v_fmac_f32_e64 v0, s39, s23                                // 0000000018F0: D52B0000 00002E27
	s_load_b128 s[36:39], s[6:7], 0x268                        // 0000000018F8: F4080903 F8000268
	s_load_b256 s[16:23], s[8:9], 0x80                         // 000000001900: F40C0404 F8000080
	s_waitcnt lgkmcnt(0)                                       // 000000001908: BF89FC07
	v_fmac_f32_e64 v0, s1, s24                                 // 00000000190C: D52B0000 00003001
	s_load_b32 s1, s[6:7], 0x17c                               // 000000001914: F4000043 F800017C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000191C: BF870091
	v_fmac_f32_e64 v0, s40, s25                                // 000000001920: D52B0000 00003228
	v_fmac_f32_e64 v0, s41, s26                                // 000000001928: D52B0000 00003429
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001930: BF8700A1
	v_fmac_f32_e64 v0, s42, s27                                // 000000001934: D52B0000 0000362A
	s_load_b128 s[24:27], s[6:7], 0x284                        // 00000000193C: F4080603 F8000284
	v_fmac_f32_e64 v0, s43, s28                                // 000000001944: D52B0000 0000382B
	s_waitcnt lgkmcnt(0)                                       // 00000000194C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001950: BF8700A1
	v_fmac_f32_e64 v0, s1, s29                                 // 000000001954: D52B0000 00003A01
	s_load_b32 s1, s[6:7], 0x278                               // 00000000195C: F4000043 F8000278
	v_fmac_f32_e64 v0, s36, s30                                // 000000001964: D52B0000 00003C24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000196C: BF870091
	v_fmac_f32_e64 v0, s37, s31                                // 000000001970: D52B0000 00003E25
	v_fmac_f32_e64 v0, s38, s16                                // 000000001978: D52B0000 00002026
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001980: BF8700A1
	v_fmac_f32_e64 v0, s39, s17                                // 000000001984: D52B0000 00002227
	s_waitcnt lgkmcnt(0)                                       // 00000000198C: BF89FC07
	v_fmac_f32_e64 v0, s1, s18                                 // 000000001990: D52B0000 00002401
	s_load_b32 s1, s[6:7], 0x294                               // 000000001998: F4000043 F8000294
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000019A0: BF8700B1
	v_fmac_f32_e64 v0, s24, s19                                // 0000000019A4: D52B0000 00002618
	s_load_b128 s[16:19], s[6:7], 0x2a0                        // 0000000019AC: F4080403 F80002A0
	s_load_b128 s[28:31], s[8:9], 0xa0                         // 0000000019B4: F4080704 F80000A0
	v_fmac_f32_e64 v0, s25, s20                                // 0000000019BC: D52B0000 00002819
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019C4: BF870091
	v_fmac_f32_e64 v0, s26, s21                                // 0000000019C8: D52B0000 00002A1A
	v_fmac_f32_e64 v0, s27, s22                                // 0000000019D0: D52B0000 00002C1B
	s_waitcnt lgkmcnt(0)                                       // 0000000019D8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019DC: BF870001
	v_fmac_f32_e64 v0, s1, s23                                 // 0000000019E0: D52B0000 00002E01
	s_load_b32 s1, s[6:7], 0x2b0                               // 0000000019E8: F4000043 F80002B0
	s_load_b32 s10, s[8:9], 0xb0                               // 0000000019F0: F4000284 F80000B0
	s_mul_i32 s6, s15, 0xa2                                    // 0000000019F8: 9606FF0F 000000A2
	s_mul_i32 s8, s14, 27                                      // 000000001A00: 96089B0E
	s_ashr_i32 s7, s6, 31                                      // 000000001A04: 86079F06
	v_fmac_f32_e64 v0, s16, s28                                // 000000001A08: D52B0000 00003810
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001A10: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A14: BF8700A9
	s_add_u32 s6, s4, s6                                       // 000000001A18: 80060604
	s_addc_u32 s7, s5, s7                                      // 000000001A1C: 82070705
	v_fmac_f32_e64 v0, s17, s29                                // 000000001A20: D52B0000 00003A11
	s_ashr_i32 s9, s8, 31                                      // 000000001A28: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A2C: BF870099
	s_lshl_b64 s[4:5], s[8:9], 2                               // 000000001A30: 84848208
	v_fmac_f32_e64 v0, s18, s30                                // 000000001A34: D52B0000 00003C12
	s_add_u32 s4, s6, s4                                       // 000000001A3C: 80040406
	s_addc_u32 s5, s7, s5                                      // 000000001A40: 82050507
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A44: BF8700A1
	v_fmac_f32_e64 v0, s19, s31                                // 000000001A48: D52B0000 00003E13
	s_waitcnt lgkmcnt(0)                                       // 000000001A50: BF89FC07
	v_fmac_f32_e64 v0, s1, s10                                 // 000000001A54: D52B0000 00001401
	s_ashr_i32 s1, s0, 31                                      // 000000001A5C: 86019F00
	v_mov_b32_e32 v1, 0                                        // 000000001A60: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001A64: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001A68: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001A6C: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001A70: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001A74: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001A78: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001A7C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001A80: DC6A0000 00000001
	s_nop 0                                                    // 000000001A88: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A8C: BFB60003
	s_endpgm                                                   // 000000001A90: BFB00000
