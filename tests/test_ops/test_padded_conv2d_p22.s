
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4_13_30_3_3_3>:
	s_load_b128 s[36:39], s[0:1], null                         // 000000001700: F4080900 F8000000
	s_mul_hi_i32 s2, s13, 0x88888889                           // 000000001708: 9702FF0D 88888889
	s_mul_i32 s8, s15, 0x39c                                   // 000000001710: 9608FF0F 0000039C
	s_add_i32 s2, s2, s13                                      // 000000001718: 81020D02
	s_mov_b32 s33, 0                                           // 00000000171C: BEA10080
	s_lshr_b32 s3, s2, 31                                      // 000000001720: 85039F02
	s_ashr_i32 s2, s2, 4                                       // 000000001724: 86028402
	s_mov_b32 s35, s33                                         // 000000001728: BEA30021
	s_add_i32 s4, s2, s3                                       // 00000000172C: 81040302
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000001730: F4040080 F8000010
	s_mul_i32 s0, s4, 28                                       // 000000001738: 96009C04
	s_mul_i32 s12, s4, 30                                      // 00000000173C: 960C9E04
	s_ashr_i32 s1, s0, 31                                      // 000000001740: 86019F00
	s_sub_i32 s34, s13, s12                                    // 000000001744: 81A20C0D
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001748: 84808200
	s_mov_b32 s42, 0                                           // 00000000174C: BEAA0080
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s0, s38, s0                                      // 000000001754: 80000026
	s_addc_u32 s1, s39, s1                                     // 000000001758: 82010127
	s_ashr_i32 s9, s8, 31                                      // 00000000175C: 86099F08
	s_cmp_gt_i32 s34, 1                                        // 000000001760: BF028122
	s_cselect_b32 s7, -1, 0                                    // 000000001764: 980780C1
	s_sub_i32 s4, s13, 60                                      // 000000001768: 8184BC0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704A9
	s_cmpk_lt_u32 s4, 0x14a                                    // 000000001770: B684014A
	s_cselect_b32 s4, -1, 0                                    // 000000001774: 980480C1
	s_and_b32 s5, s4, s7                                       // 000000001778: 8B050704
	s_add_u32 s19, s0, 0xffffff18                              // 00000000177C: 8013FF00 FFFFFF18
	v_cndmask_b32_e64 v0, 0, 1, s5                             // 000000001784: D5010000 00150280
	s_addc_u32 s18, s1, -1                                     // 00000000178C: 8212C101
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001790: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001794: BF870099
	s_add_u32 s11, s19, s8                                     // 000000001798: 800B0813
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 00000000179C: D44D0000 00020081
	s_addc_u32 s10, s18, s9                                    // 0000000017A4: 820A0912
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000017A8: 916A057E
	s_cbranch_vccnz 6                                          // 0000000017AC: BFA40006 <r_4_4_13_30_3_3_3+0xc8>
	s_lshl_b64 s[16:17], s[34:35], 2                           // 0000000017B0: 84908222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B4: BF870009
	s_add_u32 s16, s11, s16                                    // 0000000017B8: 8010100B
	s_addc_u32 s17, s10, s17                                   // 0000000017BC: 8211110A
	s_load_b32 s42, s[16:17], null                             // 0000000017C0: F4000A88 F8000000
	s_mul_i32 s16, s14, 27                                     // 0000000017C8: 96109B0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017CC: BF870499
	s_ashr_i32 s17, s16, 31                                    // 0000000017D0: 86119F10
	s_lshl_b64 s[16:17], s[16:17], 2                           // 0000000017D4: 84908210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017D8: BF8704B9
	s_add_u32 s38, s2, s16                                     // 0000000017DC: 80261002
	s_addc_u32 s39, s3, s17                                    // 0000000017E0: 82271103
	s_add_i32 s1, s34, -1                                      // 0000000017E4: 8101C122
	s_cmp_lt_u32 s1, 28                                        // 0000000017E8: BF0A9C01
	s_cselect_b32 s3, -1, 0                                    // 0000000017EC: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017F0: BF870499
	s_and_b32 s2, s4, s3                                       // 0000000017F4: 8B020304
	v_cndmask_b32_e64 v0, 0, 1, s2                             // 0000000017F8: D5010000 00090280
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000001800: 916A027E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001804: BF870001
	v_cmp_ne_u32_e64 s1, 1, v0                                 // 000000001808: D44D0001 00020081
	s_cbranch_vccnz 8                                          // 000000001810: BFA40008 <r_4_4_13_30_3_3_3+0x134>
	s_bfe_i64 s[16:17], s[34:35], 0x200000                     // 000000001814: 9490FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000181C: BF870499
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000001820: 84908210
	s_add_u32 s16, s11, s16                                    // 000000001824: 8010100B
	s_addc_u32 s17, s10, s17                                   // 000000001828: 8211110A
	s_load_b32 s33, s[16:17], 0x4                              // 00000000182C: F4000848 F8000004
	s_add_i32 s40, s34, 2                                      // 000000001834: 81288222
	s_cmp_lt_u32 s34, 28                                       // 000000001838: BF0A9C22
	s_mov_b32 s41, 0                                           // 00000000183C: BEA90080
	s_cselect_b32 s20, -1, 0                                   // 000000001840: 981480C1
	s_mov_b32 s43, s41                                         // 000000001844: BEAB0029
	s_and_b32 s4, s4, s20                                      // 000000001848: 8B041404
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000184C: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s4                             // 000000001850: D5010000 00110280
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 000000001858: 916A047E
	v_cmp_ne_u32_e64 s2, 1, v0                                 // 00000000185C: D44D0002 00020081
	s_cbranch_vccnz 6                                          // 000000001864: BFA40006 <r_4_4_13_30_3_3_3+0x180>
	s_lshl_b64 s[4:5], s[40:41], 2                             // 000000001868: 84848228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000186C: BF870009
	s_add_u32 s4, s11, s4                                      // 000000001870: 8004040B
	s_addc_u32 s5, s10, s5                                     // 000000001874: 8205050A
	s_load_b32 s43, s[4:5], null                               // 000000001878: F4000AC2 F8000000
	s_sub_i32 s4, s13, 30                                      // 000000001880: 81849E0D
	s_mov_b32 s44, s41                                         // 000000001884: BEAC0029
	s_cmpk_lt_u32 s4, 0x14a                                    // 000000001888: B684014A
	s_cselect_b32 s6, -1, 0                                    // 00000000188C: 980680C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001890: BF870009
	s_and_b32 s5, s6, s7                                       // 000000001894: 8B050706
	s_add_u32 s16, s19, s8                                     // 000000001898: 80100813
	v_cndmask_b32_e64 v0, 0, 1, s5                             // 00000000189C: D5010000 00150280
	s_addc_u32 s21, s18, s9                                    // 0000000018A4: 82150912
	s_add_u32 s17, s16, 0x70                                   // 0000000018A8: 8011FF10 00000070
	s_addc_u32 s16, s21, 0                                     // 0000000018B0: 82108015
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000018B4: 916A057E
	v_cmp_ne_u32_e64 s4, 1, v0                                 // 0000000018B8: D44D0004 00020081
	s_cbranch_vccnz 6                                          // 0000000018C0: BFA40006 <r_4_4_13_30_3_3_3+0x1dc>
	s_lshl_b64 s[22:23], s[34:35], 2                           // 0000000018C4: 84968222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018C8: BF870009
	s_add_u32 s22, s17, s22                                    // 0000000018CC: 80161611
	s_addc_u32 s23, s16, s23                                   // 0000000018D0: 82171710
	s_load_b32 s44, s[22:23], null                             // 0000000018D4: F4000B0B F8000000
	s_and_b32 s21, s6, s3                                      // 0000000018DC: 8B150306
	s_mov_b32 s45, 0                                           // 0000000018E0: BEAD0080
	v_cndmask_b32_e64 v0, 0, 1, s21                            // 0000000018E4: D5010000 00550280
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 0000000018EC: 916A157E
	s_mov_b32 s46, 0                                           // 0000000018F0: BEAE0080
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018F4: BF870001
	v_cmp_ne_u32_e64 s5, 1, v0                                 // 0000000018F8: D44D0005 00020081
	s_cbranch_vccnz 8                                          // 000000001900: BFA40008 <r_4_4_13_30_3_3_3+0x224>
	s_bfe_i64 s[22:23], s[34:35], 0x200000                     // 000000001904: 9496FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000190C: BF870499
	s_lshl_b64 s[22:23], s[22:23], 2                           // 000000001910: 84968216
	s_add_u32 s22, s17, s22                                    // 000000001914: 80161611
	s_addc_u32 s23, s16, s23                                   // 000000001918: 82171710
	s_load_b32 s46, s[22:23], 0x4                              // 00000000191C: F4000B8B F8000004
	s_and_b32 s21, s6, s20                                     // 000000001924: 8B151406
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001928: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s21                            // 00000000192C: D5010000 00550280
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001934: 916A157E
	v_cmp_ne_u32_e64 s6, 1, v0                                 // 000000001938: D44D0006 00020081
	s_cbranch_vccnz 6                                          // 000000001940: BFA40006 <r_4_4_13_30_3_3_3+0x25c>
	s_lshl_b64 s[22:23], s[40:41], 2                           // 000000001944: 84968228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001948: BF870009
	s_add_u32 s22, s17, s22                                    // 00000000194C: 80161611
	s_addc_u32 s23, s16, s23                                   // 000000001950: 82171710
	s_load_b32 s45, s[22:23], null                             // 000000001954: F4000B4B F8000000
	s_add_i32 s13, s13, 29                                     // 00000000195C: 810D9D0D
	s_mov_b32 s47, 0                                           // 000000001960: BEAF0080
	s_cmpk_lt_u32 s13, 0x167                                   // 000000001964: B68D0167
	s_mov_b32 s13, 0                                           // 000000001968: BE8D0080
	s_cselect_b32 s21, -1, 0                                   // 00000000196C: 981580C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001970: BF870009
	s_and_b32 s22, s21, s7                                     // 000000001974: 8B160715
	s_add_u32 s8, s19, s8                                      // 000000001978: 80080813
	v_cndmask_b32_e64 v0, 0, 1, s22                            // 00000000197C: D5010000 00590280
	s_addc_u32 s9, s18, s9                                     // 000000001984: 82090912
	s_add_u32 s49, s8, 0xe0                                    // 000000001988: 8031FF08 000000E0
	s_addc_u32 s48, s9, 0                                      // 000000001990: 82308009
	s_and_not1_b32 vcc_lo, exec_lo, s22                        // 000000001994: 916A167E
	v_cmp_ne_u32_e64 s7, 1, v0                                 // 000000001998: D44D0007 00020081
	s_cbranch_vccnz 6                                          // 0000000019A0: BFA40006 <r_4_4_13_30_3_3_3+0x2bc>
	s_lshl_b64 s[8:9], s[34:35], 2                             // 0000000019A4: 84888222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019A8: BF870009
	s_add_u32 s8, s49, s8                                      // 0000000019AC: 80080831
	s_addc_u32 s9, s48, s9                                     // 0000000019B0: 82090930
	s_load_b32 s47, s[8:9], null                               // 0000000019B4: F4000BC4 F8000000
	s_and_b32 s3, s21, s3                                      // 0000000019BC: 8B030315
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019C0: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 0000000019C4: D5010000 000D0280
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 0000000019CC: 916A037E
	v_cmp_ne_u32_e64 s8, 1, v0                                 // 0000000019D0: D44D0008 00020081
	s_cbranch_vccnz 8                                          // 0000000019D8: BFA40008 <r_4_4_13_30_3_3_3+0x2fc>
	s_bfe_i64 s[18:19], s[34:35], 0x200000                     // 0000000019DC: 9492FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000019E4: BF870499
	s_lshl_b64 s[18:19], s[18:19], 2                           // 0000000019E8: 84928212
	s_add_u32 s18, s49, s18                                    // 0000000019EC: 80121231
	s_addc_u32 s19, s48, s19                                   // 0000000019F0: 82131330
	s_load_b32 s13, s[18:19], 0x4                              // 0000000019F4: F4000349 F8000004
	s_and_b32 s9, s21, s20                                     // 0000000019FC: 8B091415
	s_mov_b32 s50, 0                                           // 000000001A00: BEB20080
	v_cndmask_b32_e64 v0, 0, 1, s9                             // 000000001A04: D5010000 00250280
	s_and_not1_b32 vcc_lo, exec_lo, s9                         // 000000001A0C: 916A097E
	s_mov_b32 s51, 0                                           // 000000001A10: BEB30080
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A14: BF870001
	v_cmp_ne_u32_e64 s3, 1, v0                                 // 000000001A18: D44D0003 00020081
	s_cbranch_vccz 177                                         // 000000001A20: BFA300B1 <r_4_4_13_30_3_3_3+0x5e8>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A24: 8B6A007E
	s_cbranch_vccz 183                                         // 000000001A28: BFA300B7 <r_4_4_13_30_3_3_3+0x608>
	s_mov_b32 s52, 0                                           // 000000001A2C: BEB40080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A30: 8B6A017E
	s_mov_b32 s53, 0                                           // 000000001A34: BEB50080
	s_cbranch_vccz 189                                         // 000000001A38: BFA300BD <r_4_4_13_30_3_3_3+0x630>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001A3C: 8B6A027E
	s_cbranch_vccz 197                                         // 000000001A40: BFA300C5 <r_4_4_13_30_3_3_3+0x658>
	s_mov_b32 s54, 0                                           // 000000001A44: BEB60080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001A48: 8B6A047E
	s_mov_b32 s55, 0                                           // 000000001A4C: BEB70080
	s_cbranch_vccz 203                                         // 000000001A50: BFA300CB <r_4_4_13_30_3_3_3+0x680>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001A54: 8B6A057E
	s_cbranch_vccz 209                                         // 000000001A58: BFA300D1 <r_4_4_13_30_3_3_3+0x6a0>
	s_mov_b32 s56, 0                                           // 000000001A5C: BEB80080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001A60: 8B6A067E
	s_mov_b32 s57, 0                                           // 000000001A64: BEB90080
	s_cbranch_vccz 217                                         // 000000001A68: BFA300D9 <r_4_4_13_30_3_3_3+0x6d0>
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001A6C: 8B6A077E
	s_cbranch_vccz 223                                         // 000000001A70: BFA300DF <r_4_4_13_30_3_3_3+0x6f0>
	s_mov_b32 s58, 0                                           // 000000001A74: BEBA0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001A78: 8B6A087E
	s_mov_b32 s59, 0                                           // 000000001A7C: BEBB0080
	s_cbranch_vccz 229                                         // 000000001A80: BFA300E5 <r_4_4_13_30_3_3_3+0x718>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001A84: 8B6A037E
	s_cbranch_vccz 237                                         // 000000001A88: BFA300ED <r_4_4_13_30_3_3_3+0x740>
	s_mov_b32 s60, 0                                           // 000000001A8C: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A90: 8B6A007E
	s_mov_b32 s61, 0                                           // 000000001A94: BEBD0080
	s_cbranch_vccz 243                                         // 000000001A98: BFA300F3 <r_4_4_13_30_3_3_3+0x768>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A9C: 8B6A017E
	s_cbranch_vccz 249                                         // 000000001AA0: BFA300F9 <r_4_4_13_30_3_3_3+0x788>
	s_mov_b32 s62, 0                                           // 000000001AA4: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001AA8: 8B6A027E
	s_mov_b32 s2, 0                                            // 000000001AAC: BE820080
	s_cbranch_vccz 257                                         // 000000001AB0: BFA30101 <r_4_4_13_30_3_3_3+0x7b8>
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001AB4: 8B6A047E
	s_cbranch_vccz 263                                         // 000000001AB8: BFA30107 <r_4_4_13_30_3_3_3+0x7d8>
	s_mov_b32 s63, 0                                           // 000000001ABC: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001AC0: 8B6A057E
	s_mov_b32 s64, 0                                           // 000000001AC4: BEC00080
	s_cbranch_vccz 269                                         // 000000001AC8: BFA3010D <r_4_4_13_30_3_3_3+0x800>
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001ACC: 8B6A067E
	s_cbranch_vccz 277                                         // 000000001AD0: BFA30115 <r_4_4_13_30_3_3_3+0x828>
	s_mov_b32 s65, 0                                           // 000000001AD4: BEC10080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001AD8: 8B6A077E
	s_mov_b32 s66, 0                                           // 000000001ADC: BEC20080
	s_cbranch_vccz 283                                         // 000000001AE0: BFA3011B <r_4_4_13_30_3_3_3+0x850>
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001AE4: 8B6A087E
	s_cbranch_vccnz 8                                          // 000000001AE8: BFA40008 <r_4_4_13_30_3_3_3+0x40c>
	s_bfe_i64 s[0:1], s[34:35], 0x200000                       // 000000001AEC: 9480FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001AF4: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001AF8: 84808200
	s_add_u32 s0, s49, s0                                      // 000000001AFC: 80000031
	s_addc_u32 s1, s48, s1                                     // 000000001B00: 82010130
	s_load_b32 s65, s[0:1], 0x9a4                              // 000000001B04: F4001040 F80009A4
	s_clause 0x3                                               // 000000001B0C: BF850003
	s_load_b256 s[24:31], s[38:39], null                       // 000000001B10: F40C0613 F8000000
	s_load_b256 s[16:23], s[38:39], 0x20                       // 000000001B18: F40C0413 F8000020
	s_load_b256 s[4:11], s[38:39], 0x40                        // 000000001B20: F40C0113 F8000040
	s_load_b64 s[0:1], s[38:39], 0x60                          // 000000001B28: F4040013 F8000060
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001B30: 8B6A037E
	s_mov_b32 s3, 0                                            // 000000001B34: BE830080
	s_cbranch_vccnz 6                                          // 000000001B38: BFA40006 <r_4_4_13_30_3_3_3+0x454>
	s_lshl_b64 s[40:41], s[40:41], 2                           // 000000001B3C: 84A88228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B40: BF870009
	s_add_u32 s40, s49, s40                                    // 000000001B44: 80282831
	s_addc_u32 s41, s48, s41                                   // 000000001B48: 82292930
	s_load_b32 s3, s[40:41], 0x9a0                             // 000000001B4C: F40000D4 F80009A0
	s_waitcnt lgkmcnt(0)                                       // 000000001B54: BF89FC07
	v_fma_f32 v0, s42, s24, 0                                  // 000000001B58: D6130000 0200302A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B60: BF870091
	v_fmac_f32_e64 v0, s33, s25                                // 000000001B64: D52B0000 00003221
	v_fmac_f32_e64 v0, s43, s26                                // 000000001B6C: D52B0000 0000342B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B74: BF870091
	v_fmac_f32_e64 v0, s44, s27                                // 000000001B78: D52B0000 0000362C
	v_fmac_f32_e64 v0, s46, s28                                // 000000001B80: D52B0000 0000382E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B88: BF870091
	v_fmac_f32_e64 v0, s45, s29                                // 000000001B8C: D52B0000 00003A2D
	v_fmac_f32_e64 v0, s47, s30                                // 000000001B94: D52B0000 00003C2F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001B9C: BF870091
	v_fmac_f32_e64 v0, s13, s31                                // 000000001BA0: D52B0000 00003E0D
	v_fmac_f32_e64 v0, s51, s16                                // 000000001BA8: D52B0000 00002033
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BB0: BF870091
	v_fmac_f32_e64 v0, s50, s17                                // 000000001BB4: D52B0000 00002232
	v_fmac_f32_e64 v0, s53, s18                                // 000000001BBC: D52B0000 00002435
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC4: BF870091
	v_fmac_f32_e64 v0, s52, s19                                // 000000001BC8: D52B0000 00002634
	v_fmac_f32_e64 v0, s55, s20                                // 000000001BD0: D52B0000 00002837
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BD8: BF870091
	v_fmac_f32_e64 v0, s54, s21                                // 000000001BDC: D52B0000 00002A36
	v_fmac_f32_e64 v0, s57, s22                                // 000000001BE4: D52B0000 00002C39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BEC: BF870091
	v_fmac_f32_e64 v0, s56, s23                                // 000000001BF0: D52B0000 00002E38
	v_fmac_f32_e64 v0, s59, s4                                 // 000000001BF8: D52B0000 0000083B
	s_mul_i32 s4, s15, 0x618                                   // 000000001C00: 9604FF0F 00000618
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001C08: BF8704A1
	v_fmac_f32_e64 v0, s58, s5                                 // 000000001C0C: D52B0000 00000A3A
	s_ashr_i32 s5, s4, 31                                      // 000000001C14: 86059F04
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001C18: 84848204
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001C1C: BF8700A1
	v_fmac_f32_e64 v0, s61, s6                                 // 000000001C20: D52B0000 00000C3D
	s_mul_i32 s6, s14, 0x186                                   // 000000001C28: 9606FF0E 00000186
	v_fmac_f32_e64 v0, s60, s7                                 // 000000001C30: D52B0000 00000E3C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001C38: BF8700B1
	v_fmac_f32_e64 v0, s2, s8                                  // 000000001C3C: D52B0000 00001002
	s_load_b32 s2, s[38:39], 0x68                              // 000000001C44: F4000093 F8000068
	s_add_u32 s8, s36, s4                                      // 000000001C4C: 80080424
	v_fmac_f32_e64 v0, s62, s9                                 // 000000001C50: D52B0000 0000123E
	s_addc_u32 s9, s37, s5                                     // 000000001C58: 82090525
	s_ashr_i32 s7, s6, 31                                      // 000000001C5C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C60: BF870099
	s_lshl_b64 s[4:5], s[6:7], 2                               // 000000001C64: 84848206
	v_fmac_f32_e64 v0, s64, s10                                // 000000001C68: D52B0000 00001440
	s_add_u32 s4, s8, s4                                       // 000000001C70: 80040408
	s_addc_u32 s5, s9, s5                                      // 000000001C74: 82050509
	s_ashr_i32 s13, s12, 31                                    // 000000001C78: 860D9F0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C7C: BF870091
	v_fmac_f32_e64 v0, s63, s11                                // 000000001C80: D52B0000 0000163F
	v_fmac_f32_e64 v0, s66, s0                                 // 000000001C88: D52B0000 00000042
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001C90: BF8700B1
	v_fmac_f32_e64 v0, s65, s1                                 // 000000001C94: D52B0000 00000241
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001C9C: 8480820C
	s_waitcnt lgkmcnt(0)                                       // 000000001CA0: BF89FC07
	v_fmac_f32_e64 v0, s3, s2                                  // 000000001CA4: D52B0000 00000403
	s_add_u32 s2, s4, s0                                       // 000000001CAC: 80020004
	s_addc_u32 s3, s5, s1                                      // 000000001CB0: 82030105
	s_bfe_i64 s[0:1], s[34:35], 0x200000                       // 000000001CB4: 9480FF22 00200000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001CBC: BF8704A1
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001CC0: CA140080 01000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001CC8: 84808200
	s_add_u32 s0, s2, s0                                       // 000000001CCC: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001CD0: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001CD4: DC6A0000 00000001
	s_nop 0                                                    // 000000001CDC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001CE0: BFB60003
	s_endpgm                                                   // 000000001CE4: BFB00000
	s_lshl_b64 s[18:19], s[40:41], 2                           // 000000001CE8: 84928228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CEC: BF870009
	s_add_u32 s18, s49, s18                                    // 000000001CF0: 80121231
	s_addc_u32 s19, s48, s19                                   // 000000001CF4: 82131330
	s_load_b32 s51, s[18:19], null                             // 000000001CF8: F4000CC9 F8000000
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001D00: 8B6A007E
	s_cbranch_vccnz 65353                                      // 000000001D04: BFA4FF49 <r_4_4_13_30_3_3_3+0x32c>
	s_lshl_b64 s[18:19], s[34:35], 2                           // 000000001D08: 84928222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D0C: BF870009
	s_add_u32 s18, s11, s18                                    // 000000001D10: 8012120B
	s_addc_u32 s19, s10, s19                                   // 000000001D14: 8213130A
	s_load_b32 s50, s[18:19], 0x4d0                            // 000000001D18: F4000C89 F80004D0
	s_mov_b32 s52, 0                                           // 000000001D20: BEB40080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001D24: 8B6A017E
	s_mov_b32 s53, 0                                           // 000000001D28: BEB50080
	s_cbranch_vccnz 65347                                      // 000000001D2C: BFA4FF43 <r_4_4_13_30_3_3_3+0x33c>
	s_bfe_i64 s[18:19], s[34:35], 0x200000                     // 000000001D30: 9492FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D38: BF870499
	s_lshl_b64 s[18:19], s[18:19], 2                           // 000000001D3C: 84928212
	s_add_u32 s18, s11, s18                                    // 000000001D40: 8012120B
	s_addc_u32 s19, s10, s19                                   // 000000001D44: 8213130A
	s_load_b32 s53, s[18:19], 0x4d4                            // 000000001D48: F4000D49 F80004D4
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001D50: 8B6A027E
	s_cbranch_vccnz 65339                                      // 000000001D54: BFA4FF3B <r_4_4_13_30_3_3_3+0x344>
	s_lshl_b64 s[18:19], s[40:41], 2                           // 000000001D58: 84928228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D5C: BF870009
	s_add_u32 s18, s11, s18                                    // 000000001D60: 8012120B
	s_addc_u32 s19, s10, s19                                   // 000000001D64: 8213130A
	s_load_b32 s52, s[18:19], 0x4d0                            // 000000001D68: F4000D09 F80004D0
	s_mov_b32 s54, 0                                           // 000000001D70: BEB60080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001D74: 8B6A047E
	s_mov_b32 s55, 0                                           // 000000001D78: BEB70080
	s_cbranch_vccnz 65333                                      // 000000001D7C: BFA4FF35 <r_4_4_13_30_3_3_3+0x354>
	s_lshl_b64 s[18:19], s[34:35], 2                           // 000000001D80: 84928222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D84: BF870009
	s_add_u32 s18, s17, s18                                    // 000000001D88: 80121211
	s_addc_u32 s19, s16, s19                                   // 000000001D8C: 82131310
	s_load_b32 s55, s[18:19], 0x4d0                            // 000000001D90: F4000DC9 F80004D0
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001D98: 8B6A057E
	s_cbranch_vccnz 65327                                      // 000000001D9C: BFA4FF2F <r_4_4_13_30_3_3_3+0x35c>
	s_bfe_i64 s[18:19], s[34:35], 0x200000                     // 000000001DA0: 9492FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DA8: BF870499
	s_lshl_b64 s[18:19], s[18:19], 2                           // 000000001DAC: 84928212
	s_add_u32 s18, s17, s18                                    // 000000001DB0: 80121211
	s_addc_u32 s19, s16, s19                                   // 000000001DB4: 82131310
	s_load_b32 s54, s[18:19], 0x4d4                            // 000000001DB8: F4000D89 F80004D4
	s_mov_b32 s56, 0                                           // 000000001DC0: BEB80080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001DC4: 8B6A067E
	s_mov_b32 s57, 0                                           // 000000001DC8: BEB90080
	s_cbranch_vccnz 65319                                      // 000000001DCC: BFA4FF27 <r_4_4_13_30_3_3_3+0x36c>
	s_lshl_b64 s[18:19], s[40:41], 2                           // 000000001DD0: 84928228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DD4: BF870009
	s_add_u32 s18, s17, s18                                    // 000000001DD8: 80121211
	s_addc_u32 s19, s16, s19                                   // 000000001DDC: 82131310
	s_load_b32 s57, s[18:19], 0x4d0                            // 000000001DE0: F4000E49 F80004D0
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001DE8: 8B6A077E
	s_cbranch_vccnz 65313                                      // 000000001DEC: BFA4FF21 <r_4_4_13_30_3_3_3+0x374>
	s_lshl_b64 s[18:19], s[34:35], 2                           // 000000001DF0: 84928222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DF4: BF870009
	s_add_u32 s18, s49, s18                                    // 000000001DF8: 80121231
	s_addc_u32 s19, s48, s19                                   // 000000001DFC: 82131330
	s_load_b32 s56, s[18:19], 0x4d0                            // 000000001E00: F4000E09 F80004D0
	s_mov_b32 s58, 0                                           // 000000001E08: BEBA0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001E0C: 8B6A087E
	s_mov_b32 s59, 0                                           // 000000001E10: BEBB0080
	s_cbranch_vccnz 65307                                      // 000000001E14: BFA4FF1B <r_4_4_13_30_3_3_3+0x384>
	s_bfe_i64 s[18:19], s[34:35], 0x200000                     // 000000001E18: 9492FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E20: BF870499
	s_lshl_b64 s[18:19], s[18:19], 2                           // 000000001E24: 84928212
	s_add_u32 s18, s49, s18                                    // 000000001E28: 80121231
	s_addc_u32 s19, s48, s19                                   // 000000001E2C: 82131330
	s_load_b32 s59, s[18:19], 0x4d4                            // 000000001E30: F4000EC9 F80004D4
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001E38: 8B6A037E
	s_cbranch_vccnz 65299                                      // 000000001E3C: BFA4FF13 <r_4_4_13_30_3_3_3+0x38c>
	s_lshl_b64 s[18:19], s[40:41], 2                           // 000000001E40: 84928228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E44: BF870009
	s_add_u32 s18, s49, s18                                    // 000000001E48: 80121231
	s_addc_u32 s19, s48, s19                                   // 000000001E4C: 82131330
	s_load_b32 s58, s[18:19], 0x4d0                            // 000000001E50: F4000E89 F80004D0
	s_mov_b32 s60, 0                                           // 000000001E58: BEBC0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001E5C: 8B6A007E
	s_mov_b32 s61, 0                                           // 000000001E60: BEBD0080
	s_cbranch_vccnz 65293                                      // 000000001E64: BFA4FF0D <r_4_4_13_30_3_3_3+0x39c>
	s_lshl_b64 s[18:19], s[34:35], 2                           // 000000001E68: 84928222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E6C: BF870009
	s_add_u32 s18, s11, s18                                    // 000000001E70: 8012120B
	s_addc_u32 s19, s10, s19                                   // 000000001E74: 8213130A
	s_load_b32 s61, s[18:19], 0x9a0                            // 000000001E78: F4000F49 F80009A0
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001E80: 8B6A017E
	s_cbranch_vccnz 65287                                      // 000000001E84: BFA4FF07 <r_4_4_13_30_3_3_3+0x3a4>
	s_bfe_i64 s[0:1], s[34:35], 0x200000                       // 000000001E88: 9480FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001E90: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001E94: 84808200
	s_add_u32 s0, s11, s0                                      // 000000001E98: 8000000B
	s_addc_u32 s1, s10, s1                                     // 000000001E9C: 8201010A
	s_load_b32 s60, s[0:1], 0x9a4                              // 000000001EA0: F4000F00 F80009A4
	s_mov_b32 s62, 0                                           // 000000001EA8: BEBE0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001EAC: 8B6A027E
	s_mov_b32 s2, 0                                            // 000000001EB0: BE820080
	s_cbranch_vccnz 65279                                      // 000000001EB4: BFA4FEFF <r_4_4_13_30_3_3_3+0x3b4>
	s_lshl_b64 s[0:1], s[40:41], 2                             // 000000001EB8: 84808228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EBC: BF870009
	s_add_u32 s0, s11, s0                                      // 000000001EC0: 8000000B
	s_addc_u32 s1, s10, s1                                     // 000000001EC4: 8201010A
	s_load_b32 s2, s[0:1], 0x9a0                               // 000000001EC8: F4000080 F80009A0
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001ED0: 8B6A047E
	s_cbranch_vccnz 65273                                      // 000000001ED4: BFA4FEF9 <r_4_4_13_30_3_3_3+0x3bc>
	s_lshl_b64 s[0:1], s[34:35], 2                             // 000000001ED8: 84808222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EDC: BF870009
	s_add_u32 s0, s17, s0                                      // 000000001EE0: 80000011
	s_addc_u32 s1, s16, s1                                     // 000000001EE4: 82010110
	s_load_b32 s62, s[0:1], 0x9a0                              // 000000001EE8: F4000F80 F80009A0
	s_mov_b32 s63, 0                                           // 000000001EF0: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001EF4: 8B6A057E
	s_mov_b32 s64, 0                                           // 000000001EF8: BEC00080
	s_cbranch_vccnz 65267                                      // 000000001EFC: BFA4FEF3 <r_4_4_13_30_3_3_3+0x3cc>
	s_bfe_i64 s[0:1], s[34:35], 0x200000                       // 000000001F00: 9480FF22 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001F08: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001F0C: 84808200
	s_add_u32 s0, s17, s0                                      // 000000001F10: 80000011
	s_addc_u32 s1, s16, s1                                     // 000000001F14: 82010110
	s_load_b32 s64, s[0:1], 0x9a4                              // 000000001F18: F4001000 F80009A4
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001F20: 8B6A067E
	s_cbranch_vccnz 65259                                      // 000000001F24: BFA4FEEB <r_4_4_13_30_3_3_3+0x3d4>
	s_lshl_b64 s[0:1], s[40:41], 2                             // 000000001F28: 84808228
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F2C: BF870009
	s_add_u32 s0, s17, s0                                      // 000000001F30: 80000011
	s_addc_u32 s1, s16, s1                                     // 000000001F34: 82010110
	s_load_b32 s63, s[0:1], 0x9a0                              // 000000001F38: F4000FC0 F80009A0
	s_mov_b32 s65, 0                                           // 000000001F40: BEC10080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001F44: 8B6A077E
	s_mov_b32 s66, 0                                           // 000000001F48: BEC20080
	s_cbranch_vccnz 65253                                      // 000000001F4C: BFA4FEE5 <r_4_4_13_30_3_3_3+0x3e4>
	s_lshl_b64 s[0:1], s[34:35], 2                             // 000000001F50: 84808222
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F54: BF870009
	s_add_u32 s0, s49, s0                                      // 000000001F58: 80000031
	s_addc_u32 s1, s48, s1                                     // 000000001F5C: 82010130
	s_load_b32 s66, s[0:1], 0x9a0                              // 000000001F60: F4001080 F80009A0
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001F68: 8B6A087E
	s_cbranch_vccz 65247                                       // 000000001F6C: BFA3FEDF <r_4_4_13_30_3_3_3+0x3ec>
	s_branch 65254                                             // 000000001F70: BFA0FEE6 <r_4_4_13_30_3_3_3+0x40c>
