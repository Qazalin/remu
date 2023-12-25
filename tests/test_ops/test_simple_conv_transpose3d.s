
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_4_11_11_11_4_3_3_3>:
	s_mul_hi_i32 s2, s13, 0x2e8ba2e9                           // 000000001700: 9702FF0D 2E8BA2E9
	s_mul_i32 s10, s15, 0xb64                                  // 000000001708: 960AFF0F 00000B64
	s_lshr_b32 s3, s2, 31                                      // 000000001710: 85039F02
	s_ashr_i32 s2, s2, 1                                       // 000000001714: 86028102
	s_ashr_i32 s11, s10, 31                                    // 000000001718: 860B9F0A
	s_add_i32 s3, s2, s3                                       // 00000000171C: 81030302
	s_clause 0x1                                               // 000000001720: BF850001
	s_load_b128 s[60:63], s[0:1], null                         // 000000001724: F4080F00 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000172C: F4040000 F8000010
	s_mul_hi_i32 s2, s3, 0x2e8ba2e9                            // 000000001734: 9702FF03 2E8BA2E9
	s_mul_i32 s4, s3, 11                                       // 00000000173C: 96048B03
	s_lshr_b32 s5, s2, 31                                      // 000000001740: 85059F02
	s_ashr_i32 s6, s2, 1                                       // 000000001744: 86068102
	s_sub_i32 s2, s13, s4                                      // 000000001748: 8182040D
	s_add_i32 s4, s6, s5                                       // 00000000174C: 81040506
	s_mul_hi_i32 s5, s13, 0x43b3d5b                            // 000000001750: 9705FF0D 043B3D5B
	s_mul_i32 s4, s4, 11                                       // 000000001758: 96048B04
	s_lshr_b32 s6, s5, 31                                      // 00000000175C: 85069F05
	s_ashr_i32 s27, s5, 1                                      // 000000001760: 861B8105
	s_sub_i32 s26, s3, s4                                      // 000000001764: 819A0403
	s_add_i32 s27, s27, s6                                     // 000000001768: 811B061B
	s_mul_i32 s4, s14, 27                                      // 00000000176C: 96049B0E
	s_mul_i32 s6, s26, 9                                       // 000000001770: 9606891A
	s_mul_i32 s8, s27, 0x51                                    // 000000001774: 9608FF1B 00000051
	s_ashr_i32 s5, s4, 31                                      // 00000000177C: 86059F04
	s_ashr_i32 s7, s6, 31                                      // 000000001780: 86079F06
	s_ashr_i32 s9, s8, 31                                      // 000000001784: 86099F08
	s_cmp_gt_i32 s26, 1                                        // 000000001788: BF02811A
	v_mov_b32_e32 v1, 0                                        // 00000000178C: 7E020280
	s_cselect_b32 s3, -1, 0                                    // 000000001790: 980380C1
	s_cmp_gt_i32 s2, 1                                         // 000000001794: BF028102
	s_cselect_b32 s16, -1, 0                                   // 000000001798: 981080C1
	s_add_i32 s12, s13, 0xffffff0e                             // 00000000179C: 810CFF0D FFFFFF0E
	s_and_b32 s17, s3, s16                                     // 0000000017A4: 8B111003
	s_cmpk_lt_u32 s12, 0x441                                   // 0000000017A8: B68C0441
	s_cselect_b32 s18, -1, 0                                   // 0000000017AC: 981280C1
	s_add_i32 s12, s2, -1                                      // 0000000017B0: 810CC102
	s_and_b32 s22, s18, s17                                    // 0000000017B4: 8B161112
	s_cmp_lt_u32 s12, 9                                        // 0000000017B8: BF0A890C
	s_cselect_b32 s19, -1, 0                                   // 0000000017BC: 981380C1
	s_add_i32 s12, s2, 2                                       // 0000000017C0: 810C8202
	s_and_b32 s20, s3, s19                                     // 0000000017C4: 8B141303
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000017C8: BF8704C9
	s_and_b32 s23, s18, s20                                    // 0000000017CC: 8B171412
	s_cmp_lt_u32 s2, 9                                         // 0000000017D0: BF0A8902
	v_cndmask_b32_e64 v2, 0, 1, s23                            // 0000000017D4: D5010002 005D0280
	s_cselect_b32 s21, -1, 0                                   // 0000000017DC: 981580C1
	s_and_b32 s24, s3, s21                                     // 0000000017E0: 8B181503
	s_add_i32 s3, s26, -1                                      // 0000000017E4: 8103C11A
	s_and_b32 s25, s18, s24                                    // 0000000017E8: 8B191812
	s_cmp_lt_u32 s3, 9                                         // 0000000017EC: BF0A8903
	v_cndmask_b32_e64 v3, 0, 1, s25                            // 0000000017F0: D5010003 00650280
	s_cselect_b32 s3, -1, 0                                    // 0000000017F8: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017FC: BF870009
	s_and_b32 s46, s3, s16                                     // 000000001800: 8B2E1003
	s_and_b32 s47, s3, s19                                     // 000000001804: 8B2F1303
	s_and_b32 s48, s3, s21                                     // 000000001808: 8B301503
	s_and_b32 s59, s18, s46                                    // 00000000180C: 8B3B2E12
	s_and_b32 s28, s18, s47                                    // 000000001810: 8B1C2F12
	s_and_b32 s29, s18, s48                                    // 000000001814: 8B1D3012
	s_cmp_lt_u32 s26, 9                                        // 000000001818: BF0A891A
	v_cndmask_b32_e64 v4, 0, 1, s59                            // 00000000181C: D5010004 00ED0280
	s_cselect_b32 s3, -1, 0                                    // 000000001824: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001828: BF870009
	s_and_b32 s16, s3, s16                                     // 00000000182C: 8B101003
	s_and_b32 s19, s3, s19                                     // 000000001830: 8B131303
	s_and_b32 s21, s3, s21                                     // 000000001834: 8B151503
	s_add_i32 s3, s13, 0xffffff87                              // 000000001838: 8103FF0D FFFFFF87
	s_and_b32 s30, s18, s16                                    // 000000001840: 8B1E1012
	s_and_b32 s31, s18, s19                                    // 000000001844: 8B1F1312
	s_and_b32 s33, s18, s21                                    // 000000001848: 8B211512
	s_cmpk_lt_u32 s3, 0x441                                    // 00000000184C: B6830441
	s_cselect_b32 s3, -1, 0                                    // 000000001850: 980380C1
	s_addk_i32 s13, 0x78                                       // 000000001854: B78D0078
	s_and_b32 s34, s3, s17                                     // 000000001858: 8B221103
	s_and_b32 s35, s3, s20                                     // 00000000185C: 8B231403
	s_and_b32 s36, s3, s24                                     // 000000001860: 8B241803
	s_and_b32 s37, s3, s46                                     // 000000001864: 8B252E03
	s_and_b32 s38, s3, s47                                     // 000000001868: 8B262F03
	s_and_b32 s39, s3, s48                                     // 00000000186C: 8B273003
	s_and_b32 s40, s3, s16                                     // 000000001870: 8B281003
	s_and_b32 s41, s3, s19                                     // 000000001874: 8B291303
	s_and_b32 s42, s3, s21                                     // 000000001878: 8B2A1503
	s_cmpk_lt_u32 s13, 0x4b9                                   // 00000000187C: B68D04B9
	s_mov_b32 s3, 0                                            // 000000001880: BE830080
	s_cselect_b32 s18, -1, 0                                   // 000000001884: 981280C1
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001888: 84848204
	s_mov_b32 s13, s3                                          // 00000000188C: BE8D0003
	s_and_b32 s43, s18, s17                                    // 000000001890: 8B2B1112
	s_and_b32 s44, s18, s20                                    // 000000001894: 8B2C1412
	s_and_b32 s45, s18, s24                                    // 000000001898: 8B2D1812
	s_and_b32 s46, s18, s46                                    // 00000000189C: 8B2E2E12
	s_and_b32 s47, s18, s47                                    // 0000000018A0: 8B2F2F12
	s_and_b32 s48, s18, s48                                    // 0000000018A4: 8B303012
	s_and_b32 s49, s18, s16                                    // 0000000018A8: 8B311012
	s_and_b32 s50, s18, s19                                    // 0000000018AC: 8B321312
	s_and_b32 s51, s18, s21                                    // 0000000018B0: 8B331512
	s_ashr_i32 s21, s2, 31                                     // 0000000018B4: 86159F02
	s_waitcnt lgkmcnt(0)                                       // 0000000018B8: BF89FC07
	s_add_u32 s52, s0, s4                                      // 0000000018BC: 80340400
	s_addc_u32 s53, s1, s5                                     // 0000000018C0: 82350501
	s_lshl_b64 s[0:1], s[12:13], 2                             // 0000000018C4: 8480820C
	s_mov_b32 s20, s2                                          // 0000000018C8: BE940002
	s_add_u32 s54, s0, 0xfffffd28                              // 0000000018CC: 8036FF00 FFFFFD28
	s_addc_u32 s55, s1, -1                                     // 0000000018D4: 8237C101
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000018D8: 8480820A
	s_lshl_b64 s[4:5], s[8:9], 2                               // 0000000018DC: 84848208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018E0: BF8704B9
	s_add_u32 s4, s0, s4                                       // 0000000018E4: 80040400
	s_addc_u32 s5, s1, s5                                      // 0000000018E8: 82050501
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000018EC: 84808206
	s_add_u32 s0, s4, s0                                       // 0000000018F0: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000018F4: 82010105
	s_mov_b64 s[4:5], s[60:61]                                 // 0000000018F8: BE84013C
	s_add_u32 s18, s62, s0                                     // 0000000018FC: 8012003E
	v_writelane_b32 v0, s4, 0                                  // 000000001900: D7610000 00010004
	s_addc_u32 s19, s63, s1                                    // 000000001908: 8213013F
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000190C: 84808202
	v_cmp_ne_u32_e64 s2, 1, v4                                 // 000000001910: D44D0002 00020881
	s_add_u32 s56, s0, 0xfffffd28                              // 000000001918: 8038FF00 FFFFFD28
	v_writelane_b32 v0, s5, 1                                  // 000000001920: D7610000 00010205
	s_addc_u32 s57, s1, -1                                     // 000000001928: 8239C101
	s_lshl_b64 s[12:13], s[20:21], 2                           // 00000000192C: 848C8214
	v_cmp_ne_u32_e64 s0, 1, v2                                 // 000000001930: D44D0000 00020481
	v_cmp_ne_u32_e64 s1, 1, v3                                 // 000000001938: D44D0001 00020681
	v_writelane_b32 v0, s6, 2                                  // 000000001940: D7610000 00010406
	s_add_u32 s58, s12, 0xfffffd2c                             // 000000001948: 803AFF0C FFFFFD2C
	s_addc_u32 s59, s13, -1                                    // 000000001950: 823BC10D
	s_mov_b64 s[20:21], 0                                      // 000000001954: BE940180
	s_and_b32 s3, exec_lo, s22                                 // 000000001958: 8B03167E
	v_writelane_b32 v0, s7, 3                                  // 00000000195C: D7610000 00010607
	v_writelane_b32 v0, s8, 4                                  // 000000001964: D7610000 00010808
	v_writelane_b32 v0, s9, 5                                  // 00000000196C: D7610000 00010A09
	s_branch 79                                                // 000000001974: BFA0004F <r_2_4_11_11_11_4_3_3_3+0x3b4>
	s_waitcnt lgkmcnt(0)                                       // 000000001978: BF89FC07
	v_fmac_f32_e64 v1, s60, s25                                // 00000000197C: D52B0001 0000323C
	s_add_u32 s20, s20, 0x1b0                                  // 000000001984: 8014FF14 000001B0
	s_addc_u32 s21, s21, 0                                     // 00000000198C: 82158015
	s_add_u32 s18, s18, 0xb64                                  // 000000001990: 8012FF12 00000B64
	s_addc_u32 s19, s19, 0                                     // 000000001998: 82138013
	v_fmac_f32_e64 v1, s62, s24                                // 00000000199C: D52B0001 0000303E
	s_cmpk_eq_i32 s20, 0x6c0                                   // 0000000019A4: B19406C0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019A8: BF870091
	v_fmac_f32_e64 v1, s61, s11                                // 0000000019AC: D52B0001 0000163D
	v_fmac_f32_e64 v1, s64, s10                                // 0000000019B4: D52B0001 00001440
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019BC: BF870091
	v_fmac_f32_e64 v1, s63, s9                                 // 0000000019C0: D52B0001 0000123F
	v_fmac_f32_e64 v1, s66, s8                                 // 0000000019C8: D52B0001 00001042
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019D0: BF870091
	v_fmac_f32_e64 v1, s65, s7                                 // 0000000019D4: D52B0001 00000E41
	v_fmac_f32_e64 v1, s68, s6                                 // 0000000019DC: D52B0001 00000C44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000019E4: BF870091
	v_fmac_f32_e64 v1, s67, s5                                 // 0000000019E8: D52B0001 00000A43
	v_fmac_f32_e64 v1, s70, s4                                 // 0000000019F0: D52B0001 00000846
	s_load_b32 s4, s[22:23], null                              // 0000000019F8: F400010B F8000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A00: BF870091
	v_fmac_f32_e64 v1, s69, s71                                // 000000001A04: D52B0001 00008E45
	v_fmac_f32_e64 v1, s73, s74                                // 000000001A0C: D52B0001 00009449
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A14: BF870091
	v_fmac_f32_e64 v1, s72, s75                                // 000000001A18: D52B0001 00009648
	v_fmac_f32_e64 v1, s77, s78                                // 000000001A20: D52B0001 00009C4D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A28: BF870091
	v_fmac_f32_e64 v1, s76, s79                                // 000000001A2C: D52B0001 00009E4C
	v_fmac_f32_e64 v1, s81, s82                                // 000000001A34: D52B0001 0000A451
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A3C: BF870091
	v_fmac_f32_e64 v1, s80, s83                                // 000000001A40: D52B0001 0000A650
	v_fmac_f32_e64 v1, s85, s86                                // 000000001A48: D52B0001 0000AC55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A50: BF870091
	v_fmac_f32_e64 v1, s84, s87                                // 000000001A54: D52B0001 0000AE54
	v_fmac_f32_e64 v1, s89, s90                                // 000000001A5C: D52B0001 0000B459
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A64: BF870091
	v_fmac_f32_e64 v1, s88, s92                                // 000000001A68: D52B0001 0000B858
	v_fmac_f32_e64 v1, s93, s94                                // 000000001A70: D52B0001 0000BC5D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A78: BF870091
	v_fmac_f32_e64 v1, s91, s95                                // 000000001A7C: D52B0001 0000BE5B
	v_fmac_f32_e64 v1, s97, s98                                // 000000001A84: D52B0001 0000C461
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001A8C: BF870091
	v_fmac_f32_e64 v1, s96, s99                                // 000000001A90: D52B0001 0000C660
	v_fmac_f32_e64 v1, s101, s102                              // 000000001A98: D52B0001 0000CC65
	s_waitcnt lgkmcnt(0)                                       // 000000001AA0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001AA4: BF870001
	v_fmac_f32_e64 v1, s100, s4                                // 000000001AA8: D52B0001 00000864
	s_cbranch_scc1 261                                         // 000000001AB0: BFA20105 <r_2_4_11_11_11_4_3_3_3+0x7c8>
	s_mov_b32 s60, 0                                           // 000000001AB4: BEBC0080
	s_mov_b32 vcc_lo, s3                                       // 000000001AB8: BEEA0003
	s_cbranch_vccz 4                                           // 000000001ABC: BFA30004 <r_2_4_11_11_11_4_3_3_3+0x3d0>
	s_add_u32 s4, s18, s56                                     // 000000001AC0: 80043812
	s_addc_u32 s5, s19, s57                                    // 000000001AC4: 82053913
	s_load_b32 s60, s[4:5], null                               // 000000001AC8: F4000F02 F8000000
	s_add_u32 s22, s52, s20                                    // 000000001AD0: 80161434
	s_addc_u32 s23, s53, s21                                   // 000000001AD4: 82171535
	s_mov_b32 s61, 0                                           // 000000001AD8: BEBD0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001ADC: 8B6A007E
	s_mov_b32 s62, 0                                           // 000000001AE0: BEBE0080
	s_cbranch_vccz 180                                         // 000000001AE4: BFA300B4 <r_2_4_11_11_11_4_3_3_3+0x6b8>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001AE8: 8B6A017E
	s_cbranch_vccz 184                                         // 000000001AEC: BFA300B8 <r_2_4_11_11_11_4_3_3_3+0x6d0>
	s_mov_b32 s63, 0                                           // 000000001AF0: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001AF4: 8B6A027E
	s_mov_b32 s64, 0                                           // 000000001AF8: BEC00080
	s_cbranch_vccz 188                                         // 000000001AFC: BFA300BC <r_2_4_11_11_11_4_3_3_3+0x6f0>
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 000000001B00: 916A1C7E
	s_cbranch_vccz 192                                         // 000000001B04: BFA300C0 <r_2_4_11_11_11_4_3_3_3+0x708>
	s_mov_b32 s65, 0                                           // 000000001B08: BEC10080
	s_and_not1_b32 vcc_lo, exec_lo, s29                        // 000000001B0C: 916A1D7E
	s_mov_b32 s66, 0                                           // 000000001B10: BEC20080
	s_cbranch_vccz 196                                         // 000000001B14: BFA300C4 <r_2_4_11_11_11_4_3_3_3+0x728>
	s_and_not1_b32 vcc_lo, exec_lo, s30                        // 000000001B18: 916A1E7E
	s_cbranch_vccz 200                                         // 000000001B1C: BFA300C8 <r_2_4_11_11_11_4_3_3_3+0x740>
	s_mov_b32 s67, 0                                           // 000000001B20: BEC30080
	s_and_not1_b32 vcc_lo, exec_lo, s31                        // 000000001B24: 916A1F7E
	s_mov_b32 s68, 0                                           // 000000001B28: BEC40080
	s_cbranch_vccz 204                                         // 000000001B2C: BFA300CC <r_2_4_11_11_11_4_3_3_3+0x760>
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000001B30: 916A217E
	s_cbranch_vccz 208                                         // 000000001B34: BFA300D0 <r_2_4_11_11_11_4_3_3_3+0x778>
	s_mov_b32 s69, 0                                           // 000000001B38: BEC50080
	s_and_not1_b32 vcc_lo, exec_lo, s34                        // 000000001B3C: 916A227E
	s_mov_b32 s70, 0                                           // 000000001B40: BEC60080
	s_cbranch_vccz 212                                         // 000000001B44: BFA300D4 <r_2_4_11_11_11_4_3_3_3+0x798>
	s_clause 0x1                                               // 000000001B48: BF850001
	s_load_b64 s[24:25], s[22:23], 0x64                        // 000000001B4C: F404060B F8000064
	s_load_b256 s[4:11], s[22:23], 0x44                        // 000000001B54: F40C010B F8000044
	s_and_not1_b32 vcc_lo, exec_lo, s35                        // 000000001B5C: 916A237E
	s_cbranch_vccnz 4                                          // 000000001B60: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x474>
	s_add_u32 s72, s18, s58                                    // 000000001B64: 80483A12
	s_addc_u32 s73, s19, s59                                   // 000000001B68: 82493B13
	s_load_b32 s69, s[72:73], 0x144                            // 000000001B6C: F4001164 F8000144
	s_load_b32 s71, s[22:23], 0x40                             // 000000001B74: F40011CB F8000040
	s_mov_b32 s72, 0                                           // 000000001B7C: BEC80080
	s_and_not1_b32 vcc_lo, exec_lo, s36                        // 000000001B80: 916A247E
	s_mov_b32 s73, 0                                           // 000000001B84: BEC90080
	s_cbranch_vccnz 4                                          // 000000001B88: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x49c>
	s_add_u32 s74, s18, s54                                    // 000000001B8C: 804A3612
	s_addc_u32 s75, s19, s55                                   // 000000001B90: 824B3713
	s_load_b32 s73, s[74:75], 0x144                            // 000000001B94: F4001265 F8000144
	s_load_b32 s74, s[22:23], 0x3c                             // 000000001B9C: F400128B F800003C
	s_and_not1_b32 vcc_lo, exec_lo, s37                        // 000000001BA4: 916A257E
	s_cbranch_vccnz 4                                          // 000000001BA8: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x4bc>
	s_add_u32 s76, s18, s56                                    // 000000001BAC: 804C3812
	s_addc_u32 s77, s19, s57                                   // 000000001BB0: 824D3913
	s_load_b32 s72, s[76:77], 0x168                            // 000000001BB4: F4001226 F8000168
	s_load_b32 s75, s[22:23], 0x38                             // 000000001BBC: F40012CB F8000038
	s_mov_b32 s76, 0                                           // 000000001BC4: BECC0080
	s_and_not1_b32 vcc_lo, exec_lo, s38                        // 000000001BC8: 916A267E
	s_mov_b32 s77, 0                                           // 000000001BCC: BECD0080
	s_cbranch_vccnz 4                                          // 000000001BD0: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x4e4>
	s_add_u32 s78, s18, s58                                    // 000000001BD4: 804E3A12
	s_addc_u32 s79, s19, s59                                   // 000000001BD8: 824F3B13
	s_load_b32 s77, s[78:79], 0x168                            // 000000001BDC: F4001367 F8000168
	s_load_b32 s78, s[22:23], 0x34                             // 000000001BE4: F400138B F8000034
	s_and_not1_b32 vcc_lo, exec_lo, s39                        // 000000001BEC: 916A277E
	s_cbranch_vccnz 4                                          // 000000001BF0: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x504>
	s_add_u32 s80, s18, s54                                    // 000000001BF4: 80503612
	s_addc_u32 s81, s19, s55                                   // 000000001BF8: 82513713
	s_load_b32 s76, s[80:81], 0x168                            // 000000001BFC: F4001328 F8000168
	s_load_b32 s79, s[22:23], 0x30                             // 000000001C04: F40013CB F8000030
	s_mov_b32 s80, 0                                           // 000000001C0C: BED00080
	s_and_not1_b32 vcc_lo, exec_lo, s40                        // 000000001C10: 916A287E
	s_mov_b32 s81, 0                                           // 000000001C14: BED10080
	s_cbranch_vccnz 4                                          // 000000001C18: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x52c>
	s_add_u32 s82, s18, s56                                    // 000000001C1C: 80523812
	s_addc_u32 s83, s19, s57                                   // 000000001C20: 82533913
	s_load_b32 s81, s[82:83], 0x18c                            // 000000001C24: F4001469 F800018C
	s_load_b32 s82, s[22:23], 0x2c                             // 000000001C2C: F400148B F800002C
	s_and_not1_b32 vcc_lo, exec_lo, s41                        // 000000001C34: 916A297E
	s_cbranch_vccnz 4                                          // 000000001C38: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x54c>
	s_add_u32 s84, s18, s58                                    // 000000001C3C: 80543A12
	s_addc_u32 s85, s19, s59                                   // 000000001C40: 82553B13
	s_load_b32 s80, s[84:85], 0x18c                            // 000000001C44: F400142A F800018C
	s_load_b32 s83, s[22:23], 0x28                             // 000000001C4C: F40014CB F8000028
	s_mov_b32 s84, 0                                           // 000000001C54: BED40080
	s_and_not1_b32 vcc_lo, exec_lo, s42                        // 000000001C58: 916A2A7E
	s_mov_b32 s85, 0                                           // 000000001C5C: BED50080
	s_cbranch_vccnz 4                                          // 000000001C60: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x574>
	s_add_u32 s86, s18, s54                                    // 000000001C64: 80563612
	s_addc_u32 s87, s19, s55                                   // 000000001C68: 82573713
	s_load_b32 s85, s[86:87], 0x18c                            // 000000001C6C: F400156B F800018C
	s_load_b32 s86, s[22:23], 0x24                             // 000000001C74: F400158B F8000024
	s_and_not1_b32 vcc_lo, exec_lo, s43                        // 000000001C7C: 916A2B7E
	s_cbranch_vccnz 4                                          // 000000001C80: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x594>
	s_add_u32 s88, s18, s56                                    // 000000001C84: 80583812
	s_addc_u32 s89, s19, s57                                   // 000000001C88: 82593913
	s_load_b32 s84, s[88:89], 0x288                            // 000000001C8C: F400152C F8000288
	s_load_b32 s87, s[22:23], 0x20                             // 000000001C94: F40015CB F8000020
	s_mov_b32 s88, 0                                           // 000000001C9C: BED80080
	s_and_not1_b32 vcc_lo, exec_lo, s44                        // 000000001CA0: 916A2C7E
	s_mov_b32 s89, 0                                           // 000000001CA4: BED90080
	s_cbranch_vccnz 4                                          // 000000001CA8: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x5bc>
	s_add_u32 s90, s18, s58                                    // 000000001CAC: 805A3A12
	s_addc_u32 s91, s19, s59                                   // 000000001CB0: 825B3B13
	s_load_b32 s89, s[90:91], 0x288                            // 000000001CB4: F400166D F8000288
	s_load_b32 s90, s[22:23], 0x1c                             // 000000001CBC: F400168B F800001C
	s_and_not1_b32 vcc_lo, exec_lo, s45                        // 000000001CC4: 916A2D7E
	s_cbranch_vccnz 4                                          // 000000001CC8: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x5dc>
	s_add_u32 s92, s18, s54                                    // 000000001CCC: 805C3612
	s_addc_u32 s93, s19, s55                                   // 000000001CD0: 825D3713
	s_load_b32 s88, s[92:93], 0x288                            // 000000001CD4: F400162E F8000288
	s_load_b32 s92, s[22:23], 0x18                             // 000000001CDC: F400170B F8000018
	s_mov_b32 s91, 0                                           // 000000001CE4: BEDB0080
	s_and_not1_b32 vcc_lo, exec_lo, s46                        // 000000001CE8: 916A2E7E
	s_mov_b32 s93, 0                                           // 000000001CEC: BEDD0080
	s_cbranch_vccnz 4                                          // 000000001CF0: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x604>
	s_add_u32 s94, s18, s56                                    // 000000001CF4: 805E3812
	s_addc_u32 s95, s19, s57                                   // 000000001CF8: 825F3913
	s_load_b32 s93, s[94:95], 0x2ac                            // 000000001CFC: F400176F F80002AC
	s_load_b32 s94, s[22:23], 0x14                             // 000000001D04: F400178B F8000014
	s_and_not1_b32 vcc_lo, exec_lo, s47                        // 000000001D0C: 916A2F7E
	s_cbranch_vccnz 4                                          // 000000001D10: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x624>
	s_add_u32 s96, s18, s58                                    // 000000001D14: 80603A12
	s_addc_u32 s97, s19, s59                                   // 000000001D18: 82613B13
	s_load_b32 s91, s[96:97], 0x2ac                            // 000000001D1C: F40016F0 F80002AC
	s_load_b32 s95, s[22:23], 0x10                             // 000000001D24: F40017CB F8000010
	s_mov_b32 s96, 0                                           // 000000001D2C: BEE00080
	s_and_not1_b32 vcc_lo, exec_lo, s48                        // 000000001D30: 916A307E
	s_mov_b32 s97, 0                                           // 000000001D34: BEE10080
	s_cbranch_vccnz 4                                          // 000000001D38: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x64c>
	s_add_u32 s98, s18, s54                                    // 000000001D3C: 80623612
	s_addc_u32 s99, s19, s55                                   // 000000001D40: 82633713
	s_load_b32 s97, s[98:99], 0x2ac                            // 000000001D44: F4001871 F80002AC
	s_load_b32 s98, s[22:23], 0xc                              // 000000001D4C: F400188B F800000C
	s_and_not1_b32 vcc_lo, exec_lo, s49                        // 000000001D54: 916A317E
	s_cbranch_vccnz 4                                          // 000000001D58: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x66c>
	s_add_u32 s100, s18, s56                                   // 000000001D5C: 80643812
	s_addc_u32 s101, s19, s57                                  // 000000001D60: 82653913
	s_load_b32 s96, s[100:101], 0x2d0                          // 000000001D64: F4001832 F80002D0
	s_load_b32 s99, s[22:23], 0x8                              // 000000001D6C: F40018CB F8000008
	s_mov_b32 s100, 0                                          // 000000001D74: BEE40080
	s_and_not1_b32 vcc_lo, exec_lo, s50                        // 000000001D78: 916A327E
	s_mov_b32 s101, 0                                          // 000000001D7C: BEE50080
	s_cbranch_vccnz 4                                          // 000000001D80: BFA40004 <r_2_4_11_11_11_4_3_3_3+0x694>
	s_add_u32 s102, s18, s58                                   // 000000001D84: 80663A12
	s_addc_u32 s103, s19, s59                                  // 000000001D88: 82673B13
	s_load_b32 s101, s[102:103], 0x2d0                         // 000000001D8C: F4001973 F80002D0
	s_load_b32 s102, s[22:23], 0x4                             // 000000001D94: F400198B F8000004
	s_and_not1_b32 vcc_lo, exec_lo, s51                        // 000000001D9C: 916A337E
	s_cbranch_vccnz 65269                                      // 000000001DA0: BFA4FEF5 <r_2_4_11_11_11_4_3_3_3+0x278>
	s_add_u32 s16, s18, s54                                    // 000000001DA4: 80103612
	s_addc_u32 s17, s19, s55                                   // 000000001DA8: 82113713
	s_load_b32 s100, s[16:17], 0x2d0                           // 000000001DAC: F4001908 F80002D0
	s_branch 65264                                             // 000000001DB4: BFA0FEF0 <r_2_4_11_11_11_4_3_3_3+0x278>
	s_add_u32 s4, s18, s58                                     // 000000001DB8: 80043A12
	s_addc_u32 s5, s19, s59                                    // 000000001DBC: 82053B13
	s_load_b32 s62, s[4:5], null                               // 000000001DC0: F4000F82 F8000000
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001DC8: 8B6A017E
	s_cbranch_vccnz 65352                                      // 000000001DCC: BFA4FF48 <r_2_4_11_11_11_4_3_3_3+0x3f0>
	s_add_u32 s4, s18, s54                                     // 000000001DD0: 80043612
	s_addc_u32 s5, s19, s55                                    // 000000001DD4: 82053713
	s_load_b32 s61, s[4:5], null                               // 000000001DD8: F4000F42 F8000000
	s_mov_b32 s63, 0                                           // 000000001DE0: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001DE4: 8B6A027E
	s_mov_b32 s64, 0                                           // 000000001DE8: BEC00080
	s_cbranch_vccnz 65348                                      // 000000001DEC: BFA4FF44 <r_2_4_11_11_11_4_3_3_3+0x400>
	s_add_u32 s4, s18, s56                                     // 000000001DF0: 80043812
	s_addc_u32 s5, s19, s57                                    // 000000001DF4: 82053913
	s_load_b32 s64, s[4:5], 0x24                               // 000000001DF8: F4001002 F8000024
	s_and_not1_b32 vcc_lo, exec_lo, s28                        // 000000001E00: 916A1C7E
	s_cbranch_vccnz 65344                                      // 000000001E04: BFA4FF40 <r_2_4_11_11_11_4_3_3_3+0x408>
	s_add_u32 s4, s18, s58                                     // 000000001E08: 80043A12
	s_addc_u32 s5, s19, s59                                    // 000000001E0C: 82053B13
	s_load_b32 s63, s[4:5], 0x24                               // 000000001E10: F4000FC2 F8000024
	s_mov_b32 s65, 0                                           // 000000001E18: BEC10080
	s_and_not1_b32 vcc_lo, exec_lo, s29                        // 000000001E1C: 916A1D7E
	s_mov_b32 s66, 0                                           // 000000001E20: BEC20080
	s_cbranch_vccnz 65340                                      // 000000001E24: BFA4FF3C <r_2_4_11_11_11_4_3_3_3+0x418>
	s_add_u32 s4, s18, s54                                     // 000000001E28: 80043612
	s_addc_u32 s5, s19, s55                                    // 000000001E2C: 82053713
	s_load_b32 s66, s[4:5], 0x24                               // 000000001E30: F4001082 F8000024
	s_and_not1_b32 vcc_lo, exec_lo, s30                        // 000000001E38: 916A1E7E
	s_cbranch_vccnz 65336                                      // 000000001E3C: BFA4FF38 <r_2_4_11_11_11_4_3_3_3+0x420>
	s_add_u32 s4, s18, s56                                     // 000000001E40: 80043812
	s_addc_u32 s5, s19, s57                                    // 000000001E44: 82053913
	s_load_b32 s65, s[4:5], 0x48                               // 000000001E48: F4001042 F8000048
	s_mov_b32 s67, 0                                           // 000000001E50: BEC30080
	s_and_not1_b32 vcc_lo, exec_lo, s31                        // 000000001E54: 916A1F7E
	s_mov_b32 s68, 0                                           // 000000001E58: BEC40080
	s_cbranch_vccnz 65332                                      // 000000001E5C: BFA4FF34 <r_2_4_11_11_11_4_3_3_3+0x430>
	s_add_u32 s4, s18, s58                                     // 000000001E60: 80043A12
	s_addc_u32 s5, s19, s59                                    // 000000001E64: 82053B13
	s_load_b32 s68, s[4:5], 0x48                               // 000000001E68: F4001102 F8000048
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 000000001E70: 916A217E
	s_cbranch_vccnz 65328                                      // 000000001E74: BFA4FF30 <r_2_4_11_11_11_4_3_3_3+0x438>
	s_add_u32 s4, s18, s54                                     // 000000001E78: 80043612
	s_addc_u32 s5, s19, s55                                    // 000000001E7C: 82053713
	s_load_b32 s67, s[4:5], 0x48                               // 000000001E80: F40010C2 F8000048
	s_mov_b32 s69, 0                                           // 000000001E88: BEC50080
	s_and_not1_b32 vcc_lo, exec_lo, s34                        // 000000001E8C: 916A227E
	s_mov_b32 s70, 0                                           // 000000001E90: BEC60080
	s_cbranch_vccnz 65324                                      // 000000001E94: BFA4FF2C <r_2_4_11_11_11_4_3_3_3+0x448>
	s_add_u32 s4, s18, s56                                     // 000000001E98: 80043812
	s_addc_u32 s5, s19, s57                                    // 000000001E9C: 82053913
	s_load_b32 s70, s[4:5], 0x144                              // 000000001EA0: F4001182 F8000144
	s_clause 0x1                                               // 000000001EA8: BF850001
	s_load_b64 s[24:25], s[22:23], 0x64                        // 000000001EAC: F404060B F8000064
	s_load_b256 s[4:11], s[22:23], 0x44                        // 000000001EB4: F40C010B F8000044
	s_and_not1_b32 vcc_lo, exec_lo, s35                        // 000000001EBC: 916A237E
	s_cbranch_vccz 65320                                       // 000000001EC0: BFA3FF28 <r_2_4_11_11_11_4_3_3_3+0x464>
	s_branch 65323                                             // 000000001EC4: BFA0FF2B <r_2_4_11_11_11_4_3_3_3+0x474>
	s_mul_i32 s0, s15, 0x14cc                                  // 000000001EC8: 9600FF0F 000014CC
	v_readlane_b32 s4, v0, 0                                   // 000000001ED0: D7600004 00010100
	s_ashr_i32 s1, s0, 31                                      // 000000001ED8: 86019F00
	v_readlane_b32 s5, v0, 1                                   // 000000001EDC: D7600005 00010300
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001EE4: 84808200
	s_mul_i32 s2, s14, 0x533                                   // 000000001EE8: 9602FF0E 00000533
	s_add_u32 s4, s4, s0                                       // 000000001EF0: 80040004
	v_dual_max_f32 v1, v1, v1 :: v_dual_mov_b32 v2, 0          // 000000001EF4: CA900301 01020080
	s_addc_u32 s5, s5, s1                                      // 000000001EFC: 82050105
	s_ashr_i32 s3, s2, 31                                      // 000000001F00: 86039F02
	v_readlane_b32 s6, v0, 2                                   // 000000001F04: D7600006 00010500
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001F0C: 84808202
	s_mul_i32 s2, s27, 0x79                                    // 000000001F10: 9602FF1B 00000079
	s_add_u32 s4, s4, s0                                       // 000000001F18: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001F1C: 82050105
	s_ashr_i32 s3, s2, 31                                      // 000000001F20: 86039F02
	v_max_f32_e32 v1, 0, v1                                    // 000000001F24: 20020280
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001F28: 84808202
	s_mul_i32 s2, s26, 11                                      // 000000001F2C: 96028B1A
	s_add_u32 s4, s4, s0                                       // 000000001F30: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001F34: 82050105
	s_ashr_i32 s3, s2, 31                                      // 000000001F38: 86039F02
	v_readlane_b32 s7, v0, 3                                   // 000000001F3C: D7600007 00010700
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001F44: 84808202
	v_readlane_b32 s8, v0, 4                                   // 000000001F48: D7600008 00010900
	s_add_u32 s0, s4, s0                                       // 000000001F50: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001F54: 82010105
	s_add_u32 s0, s0, s12                                      // 000000001F58: 80000C00
	s_addc_u32 s1, s1, s13                                     // 000000001F5C: 82010D01
	v_readlane_b32 s9, v0, 5                                   // 000000001F60: D7600009 00010B00
	global_store_b32 v2, v1, s[0:1]                            // 000000001F68: DC6A0000 00000102
	s_nop 0                                                    // 000000001F70: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001F74: BFB60003
	s_endpgm                                                   // 000000001F78: BFB00000
