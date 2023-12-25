
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_4_11_13_4_3_3>:
	s_load_b128 s[44:47], s[0:1], null                         // 000000001700: F4080B00 F8000000
	s_mul_hi_i32 s2, s13, 0x4ec4ec4f                           // 000000001708: 9702FF0D 4EC4EC4F
	s_mul_i32 s8, s15, 0x144                                   // 000000001710: 9608FF0F 00000144
	s_lshr_b32 s3, s2, 31                                      // 000000001718: 85039F02
	s_ashr_i32 s2, s2, 2                                       // 00000000171C: 86028202
	s_mov_b32 s33, 0                                           // 000000001720: BEA10080
	s_add_i32 s4, s2, s3                                       // 000000001724: 81040302
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000001728: F4040080 F8000010
	s_mul_i32 s0, s4, 9                                        // 000000001730: 96008904
	s_mul_i32 s12, s4, 13                                      // 000000001734: 960C8D04
	s_ashr_i32 s1, s0, 31                                      // 000000001738: 86019F00
	s_sub_i32 s34, s13, s12                                    // 00000000173C: 81A20C0D
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001740: 84808200
	s_mov_b32 s35, s33                                         // 000000001744: BEA30021
	s_mov_b32 s50, 0                                           // 000000001748: BEB20080
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	s_add_u32 s0, s46, s0                                      // 000000001750: 8000002E
	s_addc_u32 s1, s47, s1                                     // 000000001754: 8201012F
	s_ashr_i32 s9, s8, 31                                      // 000000001758: 86099F08
	s_cmp_gt_i32 s34, 3                                        // 00000000175C: BF028322
	s_cselect_b32 s7, -1, 0                                    // 000000001760: 980780C1
	s_sub_i32 s4, s13, 26                                      // 000000001764: 81849A0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001768: BF8704A9
	s_cmpk_lt_u32 s4, 0x75                                     // 00000000176C: B6840075
	s_cselect_b32 s4, -1, 0                                    // 000000001770: 980480C1
	s_and_b32 s5, s4, s7                                       // 000000001774: 8B050704
	s_add_u32 s21, s0, 0xffffffa8                              // 000000001778: 8015FF00 FFFFFFA8
	v_cndmask_b32_e64 v0, 0, 1, s5                             // 000000001780: D5010000 00150280
	s_addc_u32 s20, s1, -1                                     // 000000001788: 8214C101
	s_lshl_b64 s[8:9], s[8:9], 2                               // 00000000178C: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870099
	s_add_u32 s17, s21, s8                                     // 000000001794: 80110815
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 000000001798: D44D0000 00020081
	s_addc_u32 s16, s20, s9                                    // 0000000017A0: 82100914
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000017A4: 916A057E
	s_cbranch_vccnz 6                                          // 0000000017A8: BFA40006 <r_2_4_11_13_4_3_3+0xc4>
	s_lshl_b64 s[10:11], s[34:35], 2                           // 0000000017AC: 848A8222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B0: BF870009
	s_add_u32 s10, s17, s10                                    // 0000000017B4: 800A0A11
	s_addc_u32 s11, s16, s11                                   // 0000000017B8: 820B0B10
	s_load_b32 s50, s[10:11], null                             // 0000000017BC: F4000C85 F8000000
	s_mul_i32 s10, s14, 9                                      // 0000000017C4: 960A890E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017C8: BF870499
	s_ashr_i32 s11, s10, 31                                    // 0000000017CC: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 0000000017D0: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000017D4: BF8704D9
	s_add_u32 s10, s2, s10                                     // 0000000017D8: 800A0A02
	s_addc_u32 s11, s3, s11                                    // 0000000017DC: 820B0B03
	s_add_u32 s46, s10, 32                                     // 0000000017E0: 802EA00A
	s_addc_u32 s47, s11, 0                                     // 0000000017E4: 822F800B
	s_add_i32 s1, s34, -2                                      // 0000000017E8: 8101C222
	s_cmp_lt_u32 s1, 9                                         // 0000000017EC: BF0A8901
	s_cselect_b32 s2, -1, 0                                    // 0000000017F0: 980280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017F4: BF870499
	s_and_b32 s3, s4, s2                                       // 0000000017F8: 8B030204
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 0000000017FC: D5010000 000D0280
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001804: 916A037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001808: BF870001
	v_cmp_ne_u32_e64 s1, 1, v0                                 // 00000000180C: D44D0001 00020081
	s_cbranch_vccnz 8                                          // 000000001814: BFA40008 <r_2_4_11_13_4_3_3+0x138>
	s_bfe_i64 s[18:19], s[34:35], 0x200000                     // 000000001818: 9492FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001820: BF870499
	s_lshl_b64 s[18:19], s[18:19], 2                           // 000000001824: 84928212
	s_add_u32 s18, s17, s18                                    // 000000001828: 80121211
	s_addc_u32 s19, s16, s19                                   // 00000000182C: 82131310
	s_load_b32 s33, s[18:19], 0x8                              // 000000001830: F4000849 F8000008
	s_add_i32 s48, s34, 4                                      // 000000001838: 81308422
	s_cmp_lt_u32 s34, 9                                        // 00000000183C: BF0A8922
	s_mov_b32 s49, 0                                           // 000000001840: BEB10080
	s_cselect_b32 s22, -1, 0                                   // 000000001844: 981680C1
	s_mov_b32 s51, s49                                         // 000000001848: BEB30031
	s_and_b32 s4, s4, s22                                      // 00000000184C: 8B041604
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001850: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s4                             // 000000001854: D5010000 00110280
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 00000000185C: 916A047E
	v_cmp_ne_u32_e64 s3, 1, v0                                 // 000000001860: D44D0003 00020081
	s_cbranch_vccnz 6                                          // 000000001868: BFA40006 <r_2_4_11_13_4_3_3+0x184>
	s_lshl_b64 s[4:5], s[48:49], 2                             // 00000000186C: 84848230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001870: BF870009
	s_add_u32 s4, s17, s4                                      // 000000001874: 80040411
	s_addc_u32 s5, s16, s5                                     // 000000001878: 82050510
	s_load_b32 s51, s[4:5], null                               // 00000000187C: F4000CC2 F8000000
	s_add_i32 s4, s13, -13                                     // 000000001884: 8104CD0D
	s_mov_b32 s52, s49                                         // 000000001888: BEB40031
	s_cmpk_lt_u32 s4, 0x75                                     // 00000000188C: B6840075
	s_cselect_b32 s6, -1, 0                                    // 000000001890: 980680C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001894: BF870009
	s_and_b32 s5, s6, s7                                       // 000000001898: 8B050706
	s_add_u32 s18, s21, s8                                     // 00000000189C: 80120815
	v_cndmask_b32_e64 v0, 0, 1, s5                             // 0000000018A0: D5010000 00150280
	s_addc_u32 s23, s20, s9                                    // 0000000018A8: 82170914
	s_add_u32 s19, s18, 36                                     // 0000000018AC: 8013A412
	s_addc_u32 s18, s23, 0                                     // 0000000018B0: 82128017
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000018B4: 916A057E
	v_cmp_ne_u32_e64 s4, 1, v0                                 // 0000000018B8: D44D0004 00020081
	s_cbranch_vccnz 6                                          // 0000000018C0: BFA40006 <r_2_4_11_13_4_3_3+0x1dc>
	s_lshl_b64 s[24:25], s[34:35], 2                           // 0000000018C4: 84988222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018C8: BF870009
	s_add_u32 s24, s19, s24                                    // 0000000018CC: 80181813
	s_addc_u32 s25, s18, s25                                   // 0000000018D0: 82191912
	s_load_b32 s52, s[24:25], null                             // 0000000018D4: F4000D0C F8000000
	s_and_b32 s23, s6, s2                                      // 0000000018DC: 8B170206
	s_mov_b32 s53, 0                                           // 0000000018E0: BEB50080
	v_cndmask_b32_e64 v0, 0, 1, s23                            // 0000000018E4: D5010000 005D0280
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 0000000018EC: 916A177E
	s_mov_b32 s54, 0                                           // 0000000018F0: BEB60080
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018F4: BF870001
	v_cmp_ne_u32_e64 s5, 1, v0                                 // 0000000018F8: D44D0005 00020081
	s_cbranch_vccnz 8                                          // 000000001900: BFA40008 <r_2_4_11_13_4_3_3+0x224>
	s_bfe_i64 s[24:25], s[34:35], 0x200000                     // 000000001904: 9498FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000190C: BF870499
	s_lshl_b64 s[24:25], s[24:25], 2                           // 000000001910: 84988218
	s_add_u32 s24, s19, s24                                    // 000000001914: 80181813
	s_addc_u32 s25, s18, s25                                   // 000000001918: 82191912
	s_load_b32 s54, s[24:25], 0x8                              // 00000000191C: F4000D8C F8000008
	s_and_b32 s23, s6, s22                                     // 000000001924: 8B171606
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001928: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s23                            // 00000000192C: D5010000 005D0280
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 000000001934: 916A177E
	v_cmp_ne_u32_e64 s6, 1, v0                                 // 000000001938: D44D0006 00020081
	s_cbranch_vccnz 6                                          // 000000001940: BFA40006 <r_2_4_11_13_4_3_3+0x25c>
	s_lshl_b64 s[24:25], s[48:49], 2                           // 000000001944: 84988230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001948: BF870009
	s_add_u32 s24, s19, s24                                    // 00000000194C: 80181813
	s_addc_u32 s25, s18, s25                                   // 000000001950: 82191912
	s_load_b32 s53, s[24:25], null                             // 000000001954: F4000D4C F8000000
	s_add_i32 s13, s13, 12                                     // 00000000195C: 810D8C0D
	s_mov_b32 s55, 0                                           // 000000001960: BEB70080
	s_cmpk_lt_u32 s13, 0x81                                    // 000000001964: B68D0081
	s_mov_b32 s13, 0                                           // 000000001968: BE8D0080
	s_cselect_b32 s23, -1, 0                                   // 00000000196C: 981780C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001970: BF870009
	s_and_b32 s7, s23, s7                                      // 000000001974: 8B070717
	s_add_u32 s21, s21, s8                                     // 000000001978: 80150815
	v_cndmask_b32_e64 v0, 0, 1, s7                             // 00000000197C: D5010000 001D0280
	s_addc_u32 s9, s20, s9                                     // 000000001984: 82090914
	s_add_u32 s57, s21, 0x48                                   // 000000001988: 8039FF15 00000048
	s_addc_u32 s56, s9, 0                                      // 000000001990: 82388009
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000001994: 916A077E
	v_cmp_ne_u32_e64 s8, 1, v0                                 // 000000001998: D44D0008 00020081
	s_cbranch_vccnz 6                                          // 0000000019A0: BFA40006 <r_2_4_11_13_4_3_3+0x2bc>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 0000000019A4: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019A8: BF870009
	s_add_u32 s20, s57, s20                                    // 0000000019AC: 80141439
	s_addc_u32 s21, s56, s21                                   // 0000000019B0: 82151538
	s_load_b32 s55, s[20:21], null                             // 0000000019B4: F4000DCA F8000000
	s_and_b32 s2, s23, s2                                      // 0000000019BC: 8B020217
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019C0: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s2                             // 0000000019C4: D5010000 00090280
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000019CC: 916A027E
	v_cmp_ne_u32_e64 s7, 1, v0                                 // 0000000019D0: D44D0007 00020081
	s_cbranch_vccnz 8                                          // 0000000019D8: BFA40008 <r_2_4_11_13_4_3_3+0x2fc>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 0000000019DC: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019E4: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 0000000019E8: 84948214
	s_add_u32 s20, s57, s20                                    // 0000000019EC: 80141439
	s_addc_u32 s21, s56, s21                                   // 0000000019F0: 82151538
	s_load_b32 s13, s[20:21], 0x8                              // 0000000019F4: F400034A F8000008
	s_and_b32 s9, s23, s22                                     // 0000000019FC: 8B091617
	s_mov_b32 s58, 0                                           // 000000001A00: BEBA0080
	v_cndmask_b32_e64 v0, 0, 1, s9                             // 000000001A04: D5010000 00250280
	s_and_not1_b32 vcc_lo, exec_lo, s9                         // 000000001A0C: 916A097E
	s_mov_b32 s59, 0                                           // 000000001A10: BEBB0080
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A14: BF870001
	v_cmp_ne_u32_e64 s2, 1, v0                                 // 000000001A18: D44D0002 00020081
	s_cbranch_vccz 234                                         // 000000001A20: BFA300EA <r_2_4_11_13_4_3_3+0x6cc>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A24: 8B6A007E
	s_cbranch_vccz 240                                         // 000000001A28: BFA300F0 <r_2_4_11_13_4_3_3+0x6ec>
	s_mov_b32 s60, 0                                           // 000000001A2C: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A30: 8B6A017E
	s_mov_b32 s61, 0                                           // 000000001A34: BEBD0080
	s_cbranch_vccz 246                                         // 000000001A38: BFA300F6 <r_2_4_11_13_4_3_3+0x714>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001A3C: 8B6A037E
	s_cbranch_vccz 254                                         // 000000001A40: BFA300FE <r_2_4_11_13_4_3_3+0x73c>
	s_mov_b32 s62, 0                                           // 000000001A44: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001A48: 8B6A047E
	s_mov_b32 s63, 0                                           // 000000001A4C: BEBF0080
	s_cbranch_vccz 260                                         // 000000001A50: BFA30104 <r_2_4_11_13_4_3_3+0x764>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001A54: 8B6A057E
	s_cbranch_vccz 266                                         // 000000001A58: BFA3010A <r_2_4_11_13_4_3_3+0x784>
	s_mov_b32 s64, 0                                           // 000000001A5C: BEC00080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001A60: 8B6A067E
	s_mov_b32 s65, 0                                           // 000000001A64: BEC10080
	s_cbranch_vccz 274                                         // 000000001A68: BFA30112 <r_2_4_11_13_4_3_3+0x7b4>
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001A6C: 8B6A087E
	s_cbranch_vccz 280                                         // 000000001A70: BFA30118 <r_2_4_11_13_4_3_3+0x7d4>
	s_mov_b32 s66, 0                                           // 000000001A74: BEC20080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001A78: 8B6A077E
	s_mov_b32 s67, 0                                           // 000000001A7C: BEC30080
	s_cbranch_vccz 286                                         // 000000001A80: BFA3011E <r_2_4_11_13_4_3_3+0x7fc>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001A84: 8B6A027E
	s_cbranch_vccz 294                                         // 000000001A88: BFA30126 <r_2_4_11_13_4_3_3+0x824>
	s_mov_b32 s68, 0                                           // 000000001A8C: BEC40080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A90: 8B6A007E
	s_mov_b32 s69, 0                                           // 000000001A94: BEC50080
	s_cbranch_vccz 300                                         // 000000001A98: BFA3012C <r_2_4_11_13_4_3_3+0x84c>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A9C: 8B6A017E
	s_cbranch_vccz 306                                         // 000000001AA0: BFA30132 <r_2_4_11_13_4_3_3+0x86c>
	s_mov_b32 s70, 0                                           // 000000001AA4: BEC60080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001AA8: 8B6A037E
	s_mov_b32 s71, 0                                           // 000000001AAC: BEC70080
	s_cbranch_vccz 314                                         // 000000001AB0: BFA3013A <r_2_4_11_13_4_3_3+0x89c>
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001AB4: 8B6A047E
	s_cbranch_vccz 320                                         // 000000001AB8: BFA30140 <r_2_4_11_13_4_3_3+0x8bc>
	s_mov_b32 s72, 0                                           // 000000001ABC: BEC80080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001AC0: 8B6A057E
	s_mov_b32 s73, 0                                           // 000000001AC4: BEC90080
	s_cbranch_vccz 326                                         // 000000001AC8: BFA30146 <r_2_4_11_13_4_3_3+0x8e4>
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001ACC: 8B6A067E
	s_cbranch_vccz 334                                         // 000000001AD0: BFA3014E <r_2_4_11_13_4_3_3+0x90c>
	s_mov_b32 s74, 0                                           // 000000001AD4: BECA0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001AD8: 8B6A087E
	s_mov_b32 s75, 0                                           // 000000001ADC: BECB0080
	s_cbranch_vccz 340                                         // 000000001AE0: BFA30154 <r_2_4_11_13_4_3_3+0x934>
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001AE4: 8B6A077E
	s_cbranch_vccz 346                                         // 000000001AE8: BFA3015A <r_2_4_11_13_4_3_3+0x954>
	s_mov_b32 s76, 0                                           // 000000001AEC: BECC0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001AF0: 8B6A027E
	s_mov_b32 s77, 0                                           // 000000001AF4: BECD0080
	s_cbranch_vccz 354                                         // 000000001AF8: BFA30162 <r_2_4_11_13_4_3_3+0x984>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001AFC: 8B6A007E
	s_cbranch_vccz 360                                         // 000000001B00: BFA30168 <r_2_4_11_13_4_3_3+0x9a4>
	s_mov_b32 s0, 0                                            // 000000001B04: BE800080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001B08: 8B6A017E
	s_mov_b32 s1, 0                                            // 000000001B0C: BE810080
	s_cbranch_vccz 366                                         // 000000001B10: BFA3016E <r_2_4_11_13_4_3_3+0x9cc>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001B14: 8B6A037E
	s_cbranch_vccz 374                                         // 000000001B18: BFA30176 <r_2_4_11_13_4_3_3+0x9f4>
	s_mov_b32 s3, 0                                            // 000000001B1C: BE830080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001B20: 8B6A047E
	s_mov_b32 s78, 0                                           // 000000001B24: BECE0080
	s_cbranch_vccz 380                                         // 000000001B28: BFA3017C <r_2_4_11_13_4_3_3+0xa1c>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001B2C: 8B6A057E
	s_cbranch_vccz 386                                         // 000000001B30: BFA30182 <r_2_4_11_13_4_3_3+0xa3c>
	s_mov_b32 s79, 0                                           // 000000001B34: BECF0080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001B38: 8B6A067E
	s_mov_b32 s80, 0                                           // 000000001B3C: BED00080
	s_cbranch_vccz 394                                         // 000000001B40: BFA3018A <r_2_4_11_13_4_3_3+0xa6c>
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001B44: 8B6A087E
	s_cbranch_vccz 400                                         // 000000001B48: BFA30190 <r_2_4_11_13_4_3_3+0xa8c>
	s_mov_b32 s81, 0                                           // 000000001B4C: BED10080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001B50: 8B6A077E
	s_mov_b32 s82, 0                                           // 000000001B54: BED20080
	s_cbranch_vccnz 8                                          // 000000001B58: BFA40008 <r_2_4_11_13_4_3_3+0x47c>
	s_bfe_i64 s[4:5], s[34:35], 0x200000                       // 000000001B5C: 9484FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B64: BF870499
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001B68: 84848204
	s_add_u32 s4, s57, s4                                      // 000000001B6C: 80040439
	s_addc_u32 s5, s56, s5                                     // 000000001B70: 82050538
	s_load_b32 s82, s[4:5], 0x3d4                              // 000000001B74: F4001482 F80003D4
	s_clause 0x6                                               // 000000001B7C: BF850006
	s_load_b32 s85, s[10:11], 0x20                             // 000000001B80: F4001545 F8000020
	s_load_b256 s[36:43], s[46:47], -0x20                      // 000000001B88: F40C0917 F81FFFE0
	s_load_b32 s84, s[46:47], 0x90                             // 000000001B90: F4001517 F8000090
	s_load_b256 s[24:31], s[46:47], 0x70                       // 000000001B98: F40C0617 F8000070
	s_load_b32 s83, s[46:47], 0x120                            // 000000001BA0: F40014D7 F8000120
	s_load_b256 s[16:23], s[46:47], 0x100                      // 000000001BA8: F40C0417 F8000100
	s_load_b256 s[4:11], s[46:47], 0x194                       // 000000001BB0: F40C0117 F8000194
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001BB8: 8B6A027E
	s_cbranch_vccnz 6                                          // 000000001BBC: BFA40006 <r_2_4_11_13_4_3_3+0x4d8>
	s_lshl_b64 s[48:49], s[48:49], 2                           // 000000001BC0: 84B08230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001BC4: BF870009
	s_add_u32 s48, s57, s48                                    // 000000001BC8: 80303039
	s_addc_u32 s49, s56, s49                                   // 000000001BCC: 82313138
	s_load_b32 s81, s[48:49], 0x3cc                            // 000000001BD0: F4001458 F80003CC
	s_waitcnt lgkmcnt(0)                                       // 000000001BD8: BF89FC07
	v_fma_f32 v0, s50, s85, 0                                  // 000000001BDC: D6130000 0200AA32
	s_mul_i32 s2, s14, 0x8f                                    // 000000001BE4: 9602FF0E 0000008F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BEC: BF870091
	v_fmac_f32_e64 v0, s33, s43                                // 000000001BF0: D52B0000 00005621
	v_fmac_f32_e64 v0, s51, s42                                // 000000001BF8: D52B0000 00005433
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C00: BF870091
	v_fmac_f32_e64 v0, s52, s41                                // 000000001C04: D52B0000 00005234
	v_fmac_f32_e64 v0, s54, s40                                // 000000001C0C: D52B0000 00005036
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C14: BF870091
	v_fmac_f32_e64 v0, s53, s39                                // 000000001C18: D52B0000 00004E35
	v_fmac_f32_e64 v0, s55, s38                                // 000000001C20: D52B0000 00004C37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C28: BF870091
	v_fmac_f32_e64 v0, s13, s37                                // 000000001C2C: D52B0000 00004A0D
	v_fmac_f32_e64 v0, s59, s36                                // 000000001C34: D52B0000 0000483B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C3C: BF870091
	v_fmac_f32_e64 v0, s58, s84                                // 000000001C40: D52B0000 0000A83A
	v_fmac_f32_e64 v0, s61, s31                                // 000000001C48: D52B0000 00003E3D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C50: BF870091
	v_fmac_f32_e64 v0, s60, s30                                // 000000001C54: D52B0000 00003C3C
	v_fmac_f32_e64 v0, s63, s29                                // 000000001C5C: D52B0000 00003A3F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C64: BF870091
	v_fmac_f32_e64 v0, s62, s28                                // 000000001C68: D52B0000 0000383E
	v_fmac_f32_e64 v0, s65, s27                                // 000000001C70: D52B0000 00003641
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C78: BF870091
	v_fmac_f32_e64 v0, s64, s26                                // 000000001C7C: D52B0000 00003440
	v_fmac_f32_e64 v0, s67, s25                                // 000000001C84: D52B0000 00003243
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C8C: BF870091
	v_fmac_f32_e64 v0, s66, s24                                // 000000001C90: D52B0000 00003042
	v_fmac_f32_e64 v0, s69, s83                                // 000000001C98: D52B0000 0000A645
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CA0: BF870091
	v_fmac_f32_e64 v0, s68, s23                                // 000000001CA4: D52B0000 00002E44
	v_fmac_f32_e64 v0, s71, s22                                // 000000001CAC: D52B0000 00002C47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CB4: BF870091
	v_fmac_f32_e64 v0, s70, s21                                // 000000001CB8: D52B0000 00002A46
	v_fmac_f32_e64 v0, s73, s20                                // 000000001CC0: D52B0000 00002849
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CC8: BF870091
	v_fmac_f32_e64 v0, s72, s19                                // 000000001CCC: D52B0000 00002648
	v_fmac_f32_e64 v0, s75, s18                                // 000000001CD4: D52B0000 0000244B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CDC: BF870091
	v_fmac_f32_e64 v0, s74, s17                                // 000000001CE0: D52B0000 0000224A
	v_fmac_f32_e64 v0, s77, s16                                // 000000001CE8: D52B0000 0000204D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CF0: BF870091
	v_fmac_f32_e64 v0, s76, s11                                // 000000001CF4: D52B0000 0000164C
	v_fmac_f32_e64 v0, s1, s10                                 // 000000001CFC: D52B0000 00001401
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001D04: BF8704A1
	v_fmac_f32_e64 v0, s0, s9                                  // 000000001D08: D52B0000 00001200
	s_mul_i32 s0, s15, 0x23c                                   // 000000001D10: 9600FF0F 0000023C
	s_ashr_i32 s1, s0, 31                                      // 000000001D18: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D1C: BF870099
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001D20: 84808200
	v_fmac_f32_e64 v0, s78, s8                                 // 000000001D24: D52B0000 0000104E
	s_load_b32 s8, s[46:47], 0x190                             // 000000001D2C: F4000217 F8000190
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001D34: BF870091
	v_fmac_f32_e64 v0, s3, s7                                  // 000000001D38: D52B0000 00000E03
	v_fmac_f32_e64 v0, s80, s6                                 // 000000001D40: D52B0000 00000C50
	s_add_u32 s6, s44, s0                                      // 000000001D48: 8006002C
	s_addc_u32 s7, s45, s1                                     // 000000001D4C: 8207012D
	s_ashr_i32 s3, s2, 31                                      // 000000001D50: 86039F02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001D54: BF8704A1
	v_fmac_f32_e64 v0, s79, s5                                 // 000000001D58: D52B0000 00000A4F
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001D60: 84808202
	s_add_u32 s2, s6, s0                                       // 000000001D64: 80020006
	s_addc_u32 s3, s7, s1                                      // 000000001D68: 82030107
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001D6C: BF8704A1
	v_fmac_f32_e64 v0, s82, s4                                 // 000000001D70: D52B0000 00000852
	s_ashr_i32 s13, s12, 31                                    // 000000001D78: 860D9F0C
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001D7C: 8480820C
	s_waitcnt lgkmcnt(0)                                       // 000000001D80: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001D84: BF8700C1
	v_fmac_f32_e64 v0, s81, s8                                 // 000000001D88: D52B0000 00001051
	s_add_u32 s2, s2, s0                                       // 000000001D90: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001D94: 82030103
	s_bfe_i64 s[0:1], s[34:35], 0x200000                       // 000000001D98: 9480FF22 00200000
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001DA0: CA140080 01000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001DA8: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DAC: BF870009
	s_add_u32 s0, s2, s0                                       // 000000001DB0: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001DB4: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001DB8: DC6A0000 00000001
	s_nop 0                                                    // 000000001DC0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001DC4: BFB60003
	s_endpgm                                                   // 000000001DC8: BFB00000
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000001DCC: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DD0: BF870009
	s_add_u32 s20, s57, s20                                    // 000000001DD4: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000001DD8: 82151538
	s_load_b32 s59, s[20:21], null                             // 000000001DDC: F4000ECA F8000000
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001DE4: 8B6A007E
	s_cbranch_vccnz 65296                                      // 000000001DE8: BFA4FF10 <r_2_4_11_13_4_3_3+0x32c>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000001DEC: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DF0: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001DF4: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001DF8: 82151510
	s_load_b32 s58, s[20:21], 0x144                            // 000000001DFC: F4000E8A F8000144
	s_mov_b32 s60, 0                                           // 000000001E04: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001E08: 8B6A017E
	s_mov_b32 s61, 0                                           // 000000001E0C: BEBD0080
	s_cbranch_vccnz 65290                                      // 000000001E10: BFA4FF0A <r_2_4_11_13_4_3_3+0x33c>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000001E14: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E1C: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000001E20: 84948214
	s_add_u32 s20, s17, s20                                    // 000000001E24: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001E28: 82151510
	s_load_b32 s61, s[20:21], 0x14c                            // 000000001E2C: F4000F4A F800014C
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001E34: 8B6A037E
	s_cbranch_vccnz 65282                                      // 000000001E38: BFA4FF02 <r_2_4_11_13_4_3_3+0x344>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000001E3C: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E40: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001E44: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001E48: 82151510
	s_load_b32 s60, s[20:21], 0x144                            // 000000001E4C: F4000F0A F8000144
	s_mov_b32 s62, 0                                           // 000000001E54: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001E58: 8B6A047E
	s_mov_b32 s63, 0                                           // 000000001E5C: BEBF0080
	s_cbranch_vccnz 65276                                      // 000000001E60: BFA4FEFC <r_2_4_11_13_4_3_3+0x354>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000001E64: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E68: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001E6C: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001E70: 82151512
	s_load_b32 s63, s[20:21], 0x144                            // 000000001E74: F4000FCA F8000144
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001E7C: 8B6A057E
	s_cbranch_vccnz 65270                                      // 000000001E80: BFA4FEF6 <r_2_4_11_13_4_3_3+0x35c>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000001E84: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E8C: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000001E90: 84948214
	s_add_u32 s20, s19, s20                                    // 000000001E94: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001E98: 82151512
	s_load_b32 s62, s[20:21], 0x14c                            // 000000001E9C: F4000F8A F800014C
	s_mov_b32 s64, 0                                           // 000000001EA4: BEC00080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001EA8: 8B6A067E
	s_mov_b32 s65, 0                                           // 000000001EAC: BEC10080
	s_cbranch_vccnz 65262                                      // 000000001EB0: BFA4FEEE <r_2_4_11_13_4_3_3+0x36c>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000001EB4: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EB8: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001EBC: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001EC0: 82151512
	s_load_b32 s65, s[20:21], 0x144                            // 000000001EC4: F400104A F8000144
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001ECC: 8B6A087E
	s_cbranch_vccnz 65256                                      // 000000001ED0: BFA4FEE8 <r_2_4_11_13_4_3_3+0x374>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000001ED4: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001ED8: BF870009
	s_add_u32 s20, s57, s20                                    // 000000001EDC: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000001EE0: 82151538
	s_load_b32 s64, s[20:21], 0x144                            // 000000001EE4: F400100A F8000144
	s_mov_b32 s66, 0                                           // 000000001EEC: BEC20080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001EF0: 8B6A077E
	s_mov_b32 s67, 0                                           // 000000001EF4: BEC30080
	s_cbranch_vccnz 65250                                      // 000000001EF8: BFA4FEE2 <r_2_4_11_13_4_3_3+0x384>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000001EFC: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F04: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000001F08: 84948214
	s_add_u32 s20, s57, s20                                    // 000000001F0C: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000001F10: 82151538
	s_load_b32 s67, s[20:21], 0x14c                            // 000000001F14: F40010CA F800014C
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001F1C: 8B6A027E
	s_cbranch_vccnz 65242                                      // 000000001F20: BFA4FEDA <r_2_4_11_13_4_3_3+0x38c>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000001F24: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F28: BF870009
	s_add_u32 s20, s57, s20                                    // 000000001F2C: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000001F30: 82151538
	s_load_b32 s66, s[20:21], 0x144                            // 000000001F34: F400108A F8000144
	s_mov_b32 s68, 0                                           // 000000001F3C: BEC40080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001F40: 8B6A007E
	s_mov_b32 s69, 0                                           // 000000001F44: BEC50080
	s_cbranch_vccnz 65236                                      // 000000001F48: BFA4FED4 <r_2_4_11_13_4_3_3+0x39c>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000001F4C: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F50: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001F54: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001F58: 82151510
	s_load_b32 s69, s[20:21], 0x288                            // 000000001F5C: F400114A F8000288
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001F64: 8B6A017E
	s_cbranch_vccnz 65230                                      // 000000001F68: BFA4FECE <r_2_4_11_13_4_3_3+0x3a4>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000001F6C: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F74: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000001F78: 84948214
	s_add_u32 s20, s17, s20                                    // 000000001F7C: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001F80: 82151510
	s_load_b32 s68, s[20:21], 0x290                            // 000000001F84: F400110A F8000290
	s_mov_b32 s70, 0                                           // 000000001F8C: BEC60080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001F90: 8B6A037E
	s_mov_b32 s71, 0                                           // 000000001F94: BEC70080
	s_cbranch_vccnz 65222                                      // 000000001F98: BFA4FEC6 <r_2_4_11_13_4_3_3+0x3b4>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000001F9C: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001FA0: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001FA4: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001FA8: 82151510
	s_load_b32 s71, s[20:21], 0x288                            // 000000001FAC: F40011CA F8000288
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001FB4: 8B6A047E
	s_cbranch_vccnz 65216                                      // 000000001FB8: BFA4FEC0 <r_2_4_11_13_4_3_3+0x3bc>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000001FBC: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001FC0: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001FC4: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001FC8: 82151512
	s_load_b32 s70, s[20:21], 0x288                            // 000000001FCC: F400118A F8000288
	s_mov_b32 s72, 0                                           // 000000001FD4: BEC80080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001FD8: 8B6A057E
	s_mov_b32 s73, 0                                           // 000000001FDC: BEC90080
	s_cbranch_vccnz 65210                                      // 000000001FE0: BFA4FEBA <r_2_4_11_13_4_3_3+0x3cc>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000001FE4: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001FEC: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000001FF0: 84948214
	s_add_u32 s20, s19, s20                                    // 000000001FF4: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001FF8: 82151512
	s_load_b32 s73, s[20:21], 0x290                            // 000000001FFC: F400124A F8000290
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000002004: 8B6A067E
	s_cbranch_vccnz 65202                                      // 000000002008: BFA4FEB2 <r_2_4_11_13_4_3_3+0x3d4>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 00000000200C: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002010: BF870009
	s_add_u32 s20, s19, s20                                    // 000000002014: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000002018: 82151512
	s_load_b32 s72, s[20:21], 0x288                            // 00000000201C: F400120A F8000288
	s_mov_b32 s74, 0                                           // 000000002024: BECA0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000002028: 8B6A087E
	s_mov_b32 s75, 0                                           // 00000000202C: BECB0080
	s_cbranch_vccnz 65196                                      // 000000002030: BFA4FEAC <r_2_4_11_13_4_3_3+0x3e4>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 000000002034: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002038: BF870009
	s_add_u32 s20, s57, s20                                    // 00000000203C: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000002040: 82151538
	s_load_b32 s75, s[20:21], 0x288                            // 000000002044: F40012CA F8000288
	s_and_b32 vcc_lo, exec_lo, s7                              // 00000000204C: 8B6A077E
	s_cbranch_vccnz 65190                                      // 000000002050: BFA4FEA6 <r_2_4_11_13_4_3_3+0x3ec>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 000000002054: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000205C: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 000000002060: 84948214
	s_add_u32 s20, s57, s20                                    // 000000002064: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000002068: 82151538
	s_load_b32 s74, s[20:21], 0x290                            // 00000000206C: F400128A F8000290
	s_mov_b32 s76, 0                                           // 000000002074: BECC0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002078: 8B6A027E
	s_mov_b32 s77, 0                                           // 00000000207C: BECD0080
	s_cbranch_vccnz 65182                                      // 000000002080: BFA4FE9E <r_2_4_11_13_4_3_3+0x3fc>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 000000002084: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002088: BF870009
	s_add_u32 s20, s57, s20                                    // 00000000208C: 80141439
	s_addc_u32 s21, s56, s21                                   // 000000002090: 82151538
	s_load_b32 s77, s[20:21], 0x288                            // 000000002094: F400134A F8000288
	s_and_b32 vcc_lo, exec_lo, s0                              // 00000000209C: 8B6A007E
	s_cbranch_vccnz 65176                                      // 0000000020A0: BFA4FE98 <r_2_4_11_13_4_3_3+0x404>
	s_lshl_b64 s[20:21], s[34:35], 2                           // 0000000020A4: 84948222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000020A8: BF870009
	s_add_u32 s20, s17, s20                                    // 0000000020AC: 80141411
	s_addc_u32 s21, s16, s21                                   // 0000000020B0: 82151510
	s_load_b32 s76, s[20:21], 0x3cc                            // 0000000020B4: F400130A F80003CC
	s_mov_b32 s0, 0                                            // 0000000020BC: BE800080
	s_and_b32 vcc_lo, exec_lo, s1                              // 0000000020C0: 8B6A017E
	s_mov_b32 s1, 0                                            // 0000000020C4: BE810080
	s_cbranch_vccnz 65170                                      // 0000000020C8: BFA4FE92 <r_2_4_11_13_4_3_3+0x414>
	s_bfe_i64 s[20:21], s[34:35], 0x200000                     // 0000000020CC: 9494FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000020D4: BF870499
	s_lshl_b64 s[20:21], s[20:21], 2                           // 0000000020D8: 84948214
	s_add_u32 s20, s17, s20                                    // 0000000020DC: 80141411
	s_addc_u32 s21, s16, s21                                   // 0000000020E0: 82151510
	s_load_b32 s1, s[20:21], 0x3d4                             // 0000000020E4: F400004A F80003D4
	s_and_b32 vcc_lo, exec_lo, s3                              // 0000000020EC: 8B6A037E
	s_cbranch_vccnz 65162                                      // 0000000020F0: BFA4FE8A <r_2_4_11_13_4_3_3+0x41c>
	s_lshl_b64 s[20:21], s[48:49], 2                           // 0000000020F4: 84948230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000020F8: BF870009
	s_add_u32 s20, s17, s20                                    // 0000000020FC: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000002100: 82151510
	s_load_b32 s0, s[20:21], 0x3cc                             // 000000002104: F400000A F80003CC
	s_mov_b32 s3, 0                                            // 00000000210C: BE830080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000002110: 8B6A047E
	s_mov_b32 s78, 0                                           // 000000002114: BECE0080
	s_cbranch_vccnz 65156                                      // 000000002118: BFA4FE84 <r_2_4_11_13_4_3_3+0x42c>
	s_lshl_b64 s[16:17], s[34:35], 2                           // 00000000211C: 84908222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002120: BF870009
	s_add_u32 s16, s19, s16                                    // 000000002124: 80101013
	s_addc_u32 s17, s18, s17                                   // 000000002128: 82111112
	s_load_b32 s78, s[16:17], 0x3cc                            // 00000000212C: F4001388 F80003CC
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000002134: 8B6A057E
	s_cbranch_vccnz 65150                                      // 000000002138: BFA4FE7E <r_2_4_11_13_4_3_3+0x434>
	s_bfe_i64 s[4:5], s[34:35], 0x200000                       // 00000000213C: 9484FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000002144: BF870499
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000002148: 84848204
	s_add_u32 s4, s19, s4                                      // 00000000214C: 80040413
	s_addc_u32 s5, s18, s5                                     // 000000002150: 82050512
	s_load_b32 s3, s[4:5], 0x3d4                               // 000000002154: F40000C2 F80003D4
	s_mov_b32 s79, 0                                           // 00000000215C: BECF0080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000002160: 8B6A067E
	s_mov_b32 s80, 0                                           // 000000002164: BED00080
	s_cbranch_vccnz 65142                                      // 000000002168: BFA4FE76 <r_2_4_11_13_4_3_3+0x444>
	s_lshl_b64 s[4:5], s[48:49], 2                             // 00000000216C: 84848230
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002170: BF870009
	s_add_u32 s4, s19, s4                                      // 000000002174: 80040413
	s_addc_u32 s5, s18, s5                                     // 000000002178: 82050512
	s_load_b32 s80, s[4:5], 0x3cc                              // 00000000217C: F4001402 F80003CC
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000002184: 8B6A087E
	s_cbranch_vccnz 65136                                      // 000000002188: BFA4FE70 <r_2_4_11_13_4_3_3+0x44c>
	s_lshl_b64 s[4:5], s[34:35], 2                             // 00000000218C: 84848222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000002190: BF870009
	s_add_u32 s4, s57, s4                                      // 000000002194: 80040439
	s_addc_u32 s5, s56, s5                                     // 000000002198: 82050538
	s_load_b32 s79, s[4:5], 0x3cc                              // 00000000219C: F40013C2 F80003CC
	s_mov_b32 s81, 0                                           // 0000000021A4: BED10080
	s_and_b32 vcc_lo, exec_lo, s7                              // 0000000021A8: 8B6A077E
	s_mov_b32 s82, 0                                           // 0000000021AC: BED20080
	s_cbranch_vccz 65130                                       // 0000000021B0: BFA3FE6A <r_2_4_11_13_4_3_3+0x45c>
	s_branch 65137                                             // 0000000021B4: BFA0FE71 <r_2_4_11_13_4_3_3+0x47c>
