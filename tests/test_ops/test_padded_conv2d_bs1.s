
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_11_28_3_3_3>:
	s_load_b128 s[44:47], s[0:1], null                         // 000000001700: F4080B00 F8000000
	s_mul_i32 s2, s14, 28                                      // 000000001708: 96029C0E
	s_mov_b32 s10, s13                                         // 00000000170C: BE8A000D
	s_ashr_i32 s3, s2, 31                                      // 000000001710: 86039F02
	s_mov_b32 s9, 0                                            // 000000001714: BE890080
	s_lshl_b64 s[12:13], s[2:3], 2                             // 000000001718: 848C8202
	s_load_b64 s[2:3], s[0:1], 0x10                            // 00000000171C: F4040080 F8000010
	s_mov_b32 s33, 0                                           // 000000001724: BEA10080
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s1, s46, s12                                     // 00000000172C: 80010C2E
	s_addc_u32 s8, s47, s13                                    // 000000001730: 82080D2F
	s_add_i32 s5, s14, -1                                      // 000000001734: 8105C10E
	s_cmp_lt_i32 s14, 12                                       // 000000001738: BF048C0E
	s_cselect_b32 s4, -1, 0                                    // 00000000173C: 980480C1
	s_add_i32 s7, s10, -1                                      // 000000001740: 8107C10A
	s_cmp_lt_i32 s10, 29                                       // 000000001744: BF049D0A
	s_cselect_b32 s6, -1, 0                                    // 000000001748: 980680C1
	s_or_b32 s0, s7, s5                                        // 00000000174C: 8C000507
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001750: BF8704C9
	s_cmp_gt_i32 s0, -1                                        // 000000001754: BF02C100
	s_cselect_b32 s0, -1, 0                                    // 000000001758: 980080C1
	s_ashr_i32 s11, s10, 31                                    // 00000000175C: 860B9F0A
	s_and_b32 s0, s6, s0                                       // 000000001760: 8B000006
	s_and_b32 s18, s4, s0                                      // 000000001764: 8B120004
	s_add_u32 s17, s1, 0xffffff8c                              // 000000001768: 8011FF01 FFFFFF8C
	v_cndmask_b32_e64 v0, 0, 1, s18                            // 000000001770: D5010000 00490280
	s_addc_u32 s16, s8, -1                                     // 000000001778: 8210C108
	s_and_not1_b32 vcc_lo, exec_lo, s18                        // 00000000177C: 916A127E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001780: BF870001
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 000000001784: D44D0000 00020081
	s_cbranch_vccnz 6                                          // 00000000178C: BFA40006 <r_4_11_28_3_3_3+0xa8>
	s_lshl_b64 s[18:19], s[10:11], 2                           // 000000001790: 8492820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001794: BF870009
	s_add_u32 s18, s17, s18                                    // 000000001798: 80121211
	s_addc_u32 s19, s16, s19                                   // 00000000179C: 82131310
	s_load_b32 s33, s[18:19], null                             // 0000000017A0: F4000849 F8000000
	s_mul_i32 s18, s15, 27                                     // 0000000017A8: 96129B0F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017AC: BF870499
	s_ashr_i32 s19, s18, 31                                    // 0000000017B0: 86139F12
	s_lshl_b64 s[18:19], s[18:19], 2                           // 0000000017B4: 84928212
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000017B8: BF8704D9
	s_add_u32 s34, s2, s18                                     // 0000000017BC: 80221202
	s_addc_u32 s35, s3, s19                                    // 0000000017C0: 82231303
	s_cmp_lt_i32 s10, 28                                       // 0000000017C4: BF049C0A
	s_cselect_b32 s8, -1, 0                                    // 0000000017C8: 980880C1
	s_or_b32 s1, s10, s5                                       // 0000000017CC: 8C01050A
	s_cmp_gt_i32 s1, -1                                        // 0000000017D0: BF02C101
	s_cselect_b32 s1, -1, 0                                    // 0000000017D4: 980180C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D8: BF870499
	s_and_b32 s1, s8, s1                                       // 0000000017DC: 8B010108
	s_and_b32 s2, s4, s1                                       // 0000000017E0: 8B020104
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E4: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s2                             // 0000000017E8: D5010000 00090280
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000017F0: 916A027E
	v_cmp_ne_u32_e64 s1, 1, v0                                 // 0000000017F4: D44D0001 00020081
	s_cbranch_vccnz 6                                          // 0000000017FC: BFA40006 <r_4_11_28_3_3_3+0x118>
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001800: 8482820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001804: BF870009
	s_add_u32 s2, s17, s2                                      // 000000001808: 80020211
	s_addc_u32 s3, s16, s3                                     // 00000000180C: 82030310
	s_load_b32 s9, s[2:3], 0x4                                 // 000000001810: F4000241 F8000004
	s_add_i32 s21, s10, 1                                      // 000000001818: 8115810A
	s_cmp_lt_i32 s10, 27                                       // 00000000181C: BF049B0A
	s_mov_b32 s46, 0                                           // 000000001820: BEAE0080
	s_cselect_b32 s20, -1, 0                                   // 000000001824: 981480C1
	s_or_b32 s2, s21, s5                                       // 000000001828: 8C020515
	s_mov_b32 s47, 0                                           // 00000000182C: BEAF0080
	s_cmp_gt_i32 s2, -1                                        // 000000001830: BF02C102
	s_cselect_b32 s2, -1, 0                                    // 000000001834: 980280C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001838: BF870499
	s_and_b32 s2, s20, s2                                      // 00000000183C: 8B020214
	s_and_b32 s3, s4, s2                                       // 000000001840: 8B030204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001844: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 000000001848: D5010000 000D0280
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001850: 916A037E
	v_cmp_ne_u32_e64 s2, 1, v0                                 // 000000001854: D44D0002 00020081
	s_cbranch_vccnz 6                                          // 00000000185C: BFA40006 <r_4_11_28_3_3_3+0x178>
	s_lshl_b64 s[4:5], s[10:11], 2                             // 000000001860: 8484820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001864: BF870009
	s_add_u32 s4, s17, s4                                      // 000000001868: 80040411
	s_addc_u32 s5, s16, s5                                     // 00000000186C: 82050510
	s_load_b32 s47, s[4:5], 0x8                                // 000000001870: F4000BC2 F8000008
	s_add_i32 s22, s14, 1                                      // 000000001878: 8116810E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000187C: BF8704B9
	s_cmp_lt_i32 s22, 12                                       // 000000001880: BF048C16
	s_cselect_b32 s5, -1, 0                                    // 000000001884: 980580C1
	s_or_b32 s3, s7, s14                                       // 000000001888: 8C030E07
	s_cmp_gt_i32 s3, -1                                        // 00000000188C: BF02C103
	s_cselect_b32 s3, -1, 0                                    // 000000001890: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001894: BF870499
	s_and_b32 s3, s6, s3                                       // 000000001898: 8B030306
	s_and_b32 s4, s5, s3                                       // 00000000189C: 8B040305
	s_add_u32 s19, s17, 0x70                                   // 0000000018A0: 8013FF11 00000070
	v_cndmask_b32_e64 v0, 0, 1, s4                             // 0000000018A8: D5010000 00110280
	s_addc_u32 s18, s16, 0                                     // 0000000018B0: 82128010
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 0000000018B4: 916A047E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018B8: BF870001
	v_cmp_ne_u32_e64 s3, 1, v0                                 // 0000000018BC: D44D0003 00020081
	s_cbranch_vccnz 6                                          // 0000000018C4: BFA40006 <r_4_11_28_3_3_3+0x1e0>
	s_lshl_b64 s[24:25], s[10:11], 2                           // 0000000018C8: 8498820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018CC: BF870009
	s_add_u32 s24, s19, s24                                    // 0000000018D0: 80181813
	s_addc_u32 s25, s18, s25                                   // 0000000018D4: 82191912
	s_load_b32 s46, s[24:25], null                             // 0000000018D8: F4000B8C F8000000
	s_or_b32 s4, s10, s14                                      // 0000000018E0: 8C040E0A
	s_mov_b32 s48, 0                                           // 0000000018E4: BEB00080
	s_cmp_gt_i32 s4, -1                                        // 0000000018E8: BF02C104
	s_mov_b32 s49, 0                                           // 0000000018EC: BEB10080
	s_cselect_b32 s4, -1, 0                                    // 0000000018F0: 980480C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000018F4: BF870499
	s_and_b32 s4, s8, s4                                       // 0000000018F8: 8B040408
	s_and_b32 s23, s5, s4                                      // 0000000018FC: 8B170405
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001900: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s23                            // 000000001904: D5010000 005D0280
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 00000000190C: 916A177E
	v_cmp_ne_u32_e64 s4, 1, v0                                 // 000000001910: D44D0004 00020081
	s_cbranch_vccnz 6                                          // 000000001918: BFA40006 <r_4_11_28_3_3_3+0x234>
	s_lshl_b64 s[24:25], s[10:11], 2                           // 00000000191C: 8498820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001920: BF870009
	s_add_u32 s24, s19, s24                                    // 000000001924: 80181813
	s_addc_u32 s25, s18, s25                                   // 000000001928: 82191912
	s_load_b32 s49, s[24:25], 0x4                              // 00000000192C: F4000C4C F8000004
	s_or_b32 s23, s21, s14                                     // 000000001934: 8C170E15
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001938: BF8704A9
	s_cmp_gt_i32 s23, -1                                       // 00000000193C: BF02C117
	s_cselect_b32 s23, -1, 0                                   // 000000001940: 981780C1
	s_and_b32 s23, s20, s23                                    // 000000001944: 8B171714
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001948: BF870499
	s_and_b32 s23, s5, s23                                     // 00000000194C: 8B171705
	v_cndmask_b32_e64 v0, 0, 1, s23                            // 000000001950: D5010000 005D0280
	s_and_not1_b32 vcc_lo, exec_lo, s23                        // 000000001958: 916A177E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000195C: BF870001
	v_cmp_ne_u32_e64 s5, 1, v0                                 // 000000001960: D44D0005 00020081
	s_cbranch_vccnz 6                                          // 000000001968: BFA40006 <r_4_11_28_3_3_3+0x284>
	s_lshl_b64 s[24:25], s[10:11], 2                           // 00000000196C: 8498820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001970: BF870009
	s_add_u32 s24, s19, s24                                    // 000000001974: 80181813
	s_addc_u32 s25, s18, s25                                   // 000000001978: 82191912
	s_load_b32 s48, s[24:25], 0x8                              // 00000000197C: F4000C0C F8000008
	s_add_i32 s14, s14, 2                                      // 000000001984: 810E820E
	s_mov_b32 s50, 0                                           // 000000001988: BEB20080
	s_cmp_lt_i32 s14, 12                                       // 00000000198C: BF048C0E
	s_mov_b32 s14, 0                                           // 000000001990: BE8E0080
	s_cselect_b32 s23, -1, 0                                   // 000000001994: 981780C1
	s_or_b32 s7, s7, s22                                       // 000000001998: 8C071607
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000199C: BF8704A9
	s_cmp_gt_i32 s7, -1                                        // 0000000019A0: BF02C107
	s_cselect_b32 s7, -1, 0                                    // 0000000019A4: 980780C1
	s_and_b32 s6, s6, s7                                       // 0000000019A8: 8B060706
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000019AC: BF8700D9
	s_and_b32 s6, s23, s6                                      // 0000000019B0: 8B060617
	s_add_u32 s52, s17, 0xe0                                   // 0000000019B4: 8034FF11 000000E0
	v_cndmask_b32_e64 v0, 0, 1, s6                             // 0000000019BC: D5010000 00190280
	s_addc_u32 s51, s16, 0                                     // 0000000019C4: 82338010
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 0000000019C8: 916A067E
	v_cmp_ne_u32_e64 s7, 1, v0                                 // 0000000019CC: D44D0007 00020081
	s_cbranch_vccnz 6                                          // 0000000019D4: BFA40006 <r_4_11_28_3_3_3+0x2f0>
	s_lshl_b64 s[24:25], s[10:11], 2                           // 0000000019D8: 8498820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019DC: BF870009
	s_add_u32 s24, s52, s24                                    // 0000000019E0: 80181834
	s_addc_u32 s25, s51, s25                                   // 0000000019E4: 82191933
	s_load_b32 s50, s[24:25], null                             // 0000000019E8: F4000C8C F8000000
	s_or_b32 s6, s10, s22                                      // 0000000019F0: 8C06160A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000019F4: BF8704A9
	s_cmp_gt_i32 s6, -1                                        // 0000000019F8: BF02C106
	s_cselect_b32 s6, -1, 0                                    // 0000000019FC: 980680C1
	s_and_b32 s6, s8, s6                                       // 000000001A00: 8B060608
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A04: BF870499
	s_and_b32 s6, s23, s6                                      // 000000001A08: 8B060617
	v_cndmask_b32_e64 v0, 0, 1, s6                             // 000000001A0C: D5010000 00190280
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000001A14: 916A067E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A18: BF870001
	v_cmp_ne_u32_e64 s8, 1, v0                                 // 000000001A1C: D44D0008 00020081
	s_cbranch_vccnz 6                                          // 000000001A24: BFA40006 <r_4_11_28_3_3_3+0x340>
	s_lshl_b64 s[24:25], s[10:11], 2                           // 000000001A28: 8498820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A2C: BF870009
	s_add_u32 s24, s52, s24                                    // 000000001A30: 80181834
	s_addc_u32 s25, s51, s25                                   // 000000001A34: 82191933
	s_load_b32 s14, s[24:25], 0x4                              // 000000001A38: F400038C F8000004
	s_or_b32 s6, s21, s22                                      // 000000001A40: 8C061615
	s_mov_b32 s53, 0                                           // 000000001A44: BEB50080
	s_cmp_gt_i32 s6, -1                                        // 000000001A48: BF02C106
	s_mov_b32 s54, 0                                           // 000000001A4C: BEB60080
	s_cselect_b32 s6, -1, 0                                    // 000000001A50: 980680C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001A54: BF870499
	s_and_b32 s6, s20, s6                                      // 000000001A58: 8B060614
	s_and_b32 s20, s23, s6                                     // 000000001A5C: 8B140617
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A60: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s20                            // 000000001A64: D5010000 00510280
	s_and_not1_b32 vcc_lo, exec_lo, s20                        // 000000001A6C: 916A147E
	v_cmp_ne_u32_e64 s6, 1, v0                                 // 000000001A70: D44D0006 00020081
	s_cbranch_vccz 165                                         // 000000001A78: BFA300A5 <r_4_11_28_3_3_3+0x610>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A7C: 8B6A007E
	s_cbranch_vccz 171                                         // 000000001A80: BFA300AB <r_4_11_28_3_3_3+0x630>
	s_mov_b32 s55, 0                                           // 000000001A84: BEB70080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A88: 8B6A017E
	s_mov_b32 s56, 0                                           // 000000001A8C: BEB80080
	s_cbranch_vccz 177                                         // 000000001A90: BFA300B1 <r_4_11_28_3_3_3+0x658>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001A94: 8B6A027E
	s_cbranch_vccz 183                                         // 000000001A98: BFA300B7 <r_4_11_28_3_3_3+0x678>
	s_mov_b32 s57, 0                                           // 000000001A9C: BEB90080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001AA0: 8B6A037E
	s_mov_b32 s58, 0                                           // 000000001AA4: BEBA0080
	s_cbranch_vccz 189                                         // 000000001AA8: BFA300BD <r_4_11_28_3_3_3+0x6a0>
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001AAC: 8B6A047E
	s_cbranch_vccz 195                                         // 000000001AB0: BFA300C3 <r_4_11_28_3_3_3+0x6c0>
	s_mov_b32 s59, 0                                           // 000000001AB4: BEBB0080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001AB8: 8B6A057E
	s_mov_b32 s60, 0                                           // 000000001ABC: BEBC0080
	s_cbranch_vccz 201                                         // 000000001AC0: BFA300C9 <r_4_11_28_3_3_3+0x6e8>
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001AC4: 8B6A077E
	s_cbranch_vccz 207                                         // 000000001AC8: BFA300CF <r_4_11_28_3_3_3+0x708>
	s_mov_b32 s61, 0                                           // 000000001ACC: BEBD0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001AD0: 8B6A087E
	s_mov_b32 s62, 0                                           // 000000001AD4: BEBE0080
	s_cbranch_vccz 213                                         // 000000001AD8: BFA300D5 <r_4_11_28_3_3_3+0x730>
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001ADC: 8B6A067E
	s_cbranch_vccz 219                                         // 000000001AE0: BFA300DB <r_4_11_28_3_3_3+0x750>
	s_mov_b32 s63, 0                                           // 000000001AE4: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001AE8: 8B6A007E
	s_mov_b32 s64, 0                                           // 000000001AEC: BEC00080
	s_cbranch_vccz 225                                         // 000000001AF0: BFA300E1 <r_4_11_28_3_3_3+0x778>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001AF4: 8B6A017E
	s_cbranch_vccz 231                                         // 000000001AF8: BFA300E7 <r_4_11_28_3_3_3+0x798>
	s_mov_b32 s65, 0                                           // 000000001AFC: BEC10080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001B00: 8B6A027E
	s_mov_b32 s2, 0                                            // 000000001B04: BE820080
	s_cbranch_vccz 237                                         // 000000001B08: BFA300ED <r_4_11_28_3_3_3+0x7c0>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001B0C: 8B6A037E
	s_cbranch_vccz 243                                         // 000000001B10: BFA300F3 <r_4_11_28_3_3_3+0x7e0>
	s_mov_b32 s3, 0                                            // 000000001B14: BE830080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001B18: 8B6A047E
	s_mov_b32 s4, 0                                            // 000000001B1C: BE840080
	s_cbranch_vccz 249                                         // 000000001B20: BFA300F9 <r_4_11_28_3_3_3+0x808>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001B24: 8B6A057E
	s_cbranch_vccz 255                                         // 000000001B28: BFA300FF <r_4_11_28_3_3_3+0x828>
	s_mov_b32 s5, 0                                            // 000000001B2C: BE850080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001B30: 8B6A077E
	s_mov_b32 s7, 0                                            // 000000001B34: BE870080
	s_cbranch_vccz 261                                         // 000000001B38: BFA30105 <r_4_11_28_3_3_3+0x850>
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001B3C: 8B6A087E
	s_cbranch_vccnz 6                                          // 000000001B40: BFA40006 <r_4_11_28_3_3_3+0x45c>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001B44: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B48: BF870009
	s_add_u32 s0, s52, s0                                      // 000000001B4C: 80000034
	s_addc_u32 s1, s51, s1                                     // 000000001B50: 82010133
	s_load_b32 s5, s[0:1], 0x9a4                               // 000000001B54: F4000140 F80009A4
	s_clause 0x3                                               // 000000001B5C: BF850003
	s_load_b256 s[36:43], s[34:35], null                       // 000000001B60: F40C0911 F8000000
	s_load_b256 s[24:31], s[34:35], 0x20                       // 000000001B68: F40C0611 F8000020
	s_load_b256 s[16:23], s[34:35], 0x40                       // 000000001B70: F40C0411 F8000040
	s_load_b64 s[0:1], s[34:35], 0x60                          // 000000001B78: F4040011 F8000060
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001B80: 8B6A067E
	s_mov_b32 s6, 0                                            // 000000001B84: BE860080
	s_cbranch_vccnz 6                                          // 000000001B88: BFA40006 <r_4_11_28_3_3_3+0x4a4>
	s_lshl_b64 s[66:67], s[10:11], 2                           // 000000001B8C: 84C2820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B90: BF870009
	s_add_u32 s66, s52, s66                                    // 000000001B94: 80424234
	s_addc_u32 s67, s51, s67                                   // 000000001B98: 82434333
	s_load_b32 s6, s[66:67], 0x9a8                             // 000000001B9C: F40001A1 F80009A8
	s_waitcnt lgkmcnt(0)                                       // 000000001BA4: BF89FC07
	v_fma_f32 v0, s33, s36, 0                                  // 000000001BA8: D6130000 02004821
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BB0: BF870091
	v_fmac_f32_e64 v0, s9, s37                                 // 000000001BB4: D52B0000 00004A09
	v_fmac_f32_e64 v0, s47, s38                                // 000000001BBC: D52B0000 00004C2F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BC4: BF870091
	v_fmac_f32_e64 v0, s46, s39                                // 000000001BC8: D52B0000 00004E2E
	v_fmac_f32_e64 v0, s49, s40                                // 000000001BD0: D52B0000 00005031
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BD8: BF870091
	v_fmac_f32_e64 v0, s48, s41                                // 000000001BDC: D52B0000 00005230
	v_fmac_f32_e64 v0, s50, s42                                // 000000001BE4: D52B0000 00005432
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001BEC: BF870091
	v_fmac_f32_e64 v0, s14, s43                                // 000000001BF0: D52B0000 0000560E
	v_fmac_f32_e64 v0, s54, s24                                // 000000001BF8: D52B0000 00003036
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C00: BF870091
	v_fmac_f32_e64 v0, s53, s25                                // 000000001C04: D52B0000 00003235
	v_fmac_f32_e64 v0, s56, s26                                // 000000001C0C: D52B0000 00003438
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C14: BF870091
	v_fmac_f32_e64 v0, s55, s27                                // 000000001C18: D52B0000 00003637
	v_fmac_f32_e64 v0, s58, s28                                // 000000001C20: D52B0000 0000383A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C28: BF870091
	v_fmac_f32_e64 v0, s57, s29                                // 000000001C2C: D52B0000 00003A39
	v_fmac_f32_e64 v0, s60, s30                                // 000000001C34: D52B0000 00003C3C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C3C: BF870091
	v_fmac_f32_e64 v0, s59, s31                                // 000000001C40: D52B0000 00003E3B
	v_fmac_f32_e64 v0, s62, s16                                // 000000001C48: D52B0000 0000203E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C50: BF870091
	v_fmac_f32_e64 v0, s61, s17                                // 000000001C54: D52B0000 0000223D
	v_fmac_f32_e64 v0, s64, s18                                // 000000001C5C: D52B0000 00002440
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C64: BF870091
	v_fmac_f32_e64 v0, s63, s19                                // 000000001C68: D52B0000 0000263F
	v_fmac_f32_e64 v0, s2, s20                                 // 000000001C70: D52B0000 00002802
	s_load_b32 s2, s[34:35], 0x68                              // 000000001C78: F4000091 F8000068
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C80: BF870091
	v_fmac_f32_e64 v0, s65, s21                                // 000000001C84: D52B0000 00002A41
	v_fmac_f32_e64 v0, s4, s22                                 // 000000001C8C: D52B0000 00002C04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001C94: BF870091
	v_fmac_f32_e64 v0, s3, s23                                 // 000000001C98: D52B0000 00002E03
	v_fmac_f32_e64 v0, s7, s0                                  // 000000001CA0: D52B0000 00000007
	s_mul_i32 s0, s15, 0x134                                   // 000000001CA8: 9600FF0F 00000134
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001CB0: BF8704A1
	v_fmac_f32_e64 v0, s5, s1                                  // 000000001CB4: D52B0000 00000205
	s_ashr_i32 s1, s0, 31                                      // 000000001CBC: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001CC0: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CC4: BF870009
	s_add_u32 s0, s44, s0                                      // 000000001CC8: 8000002C
	s_waitcnt lgkmcnt(0)                                       // 000000001CCC: BF89FC07
	v_fmac_f32_e64 v0, s6, s2                                  // 000000001CD0: D52B0000 00000406
	s_addc_u32 s1, s45, s1                                     // 000000001CD8: 8201012D
	s_add_u32 s2, s0, s12                                      // 000000001CDC: 80020C00
	v_mov_b32_e32 v1, 0                                        // 000000001CE0: 7E020280
	s_addc_u32 s3, s1, s13                                     // 000000001CE4: 82030D01
	v_max_f32_e32 v0, 0, v0                                    // 000000001CE8: 20000080
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001CEC: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CF0: BF870009
	s_add_u32 s0, s2, s0                                       // 000000001CF4: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001CF8: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001CFC: DC6A0000 00000001
	s_nop 0                                                    // 000000001D04: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001D08: BFB60003
	s_endpgm                                                   // 000000001D0C: BFB00000
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001D10: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D14: BF870009
	s_add_u32 s20, s52, s20                                    // 000000001D18: 80141434
	s_addc_u32 s21, s51, s21                                   // 000000001D1C: 82151533
	s_load_b32 s54, s[20:21], 0x8                              // 000000001D20: F4000D8A F8000008
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001D28: 8B6A007E
	s_cbranch_vccnz 65365                                      // 000000001D2C: BFA4FF55 <r_4_11_28_3_3_3+0x384>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001D30: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D34: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001D38: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001D3C: 82151510
	s_load_b32 s53, s[20:21], 0x4d0                            // 000000001D40: F4000D4A F80004D0
	s_mov_b32 s55, 0                                           // 000000001D48: BEB70080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001D4C: 8B6A017E
	s_mov_b32 s56, 0                                           // 000000001D50: BEB80080
	s_cbranch_vccnz 65359                                      // 000000001D54: BFA4FF4F <r_4_11_28_3_3_3+0x394>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001D58: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D5C: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001D60: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001D64: 82151510
	s_load_b32 s56, s[20:21], 0x4d4                            // 000000001D68: F4000E0A F80004D4
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001D70: 8B6A027E
	s_cbranch_vccnz 65353                                      // 000000001D74: BFA4FF49 <r_4_11_28_3_3_3+0x39c>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001D78: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D7C: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001D80: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001D84: 82151510
	s_load_b32 s55, s[20:21], 0x4d8                            // 000000001D88: F4000DCA F80004D8
	s_mov_b32 s57, 0                                           // 000000001D90: BEB90080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001D94: 8B6A037E
	s_mov_b32 s58, 0                                           // 000000001D98: BEBA0080
	s_cbranch_vccnz 65347                                      // 000000001D9C: BFA4FF43 <r_4_11_28_3_3_3+0x3ac>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001DA0: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DA4: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001DA8: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001DAC: 82151512
	s_load_b32 s58, s[20:21], 0x4d0                            // 000000001DB0: F4000E8A F80004D0
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001DB8: 8B6A047E
	s_cbranch_vccnz 65341                                      // 000000001DBC: BFA4FF3D <r_4_11_28_3_3_3+0x3b4>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001DC0: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DC4: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001DC8: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001DCC: 82151512
	s_load_b32 s57, s[20:21], 0x4d4                            // 000000001DD0: F4000E4A F80004D4
	s_mov_b32 s59, 0                                           // 000000001DD8: BEBB0080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001DDC: 8B6A057E
	s_mov_b32 s60, 0                                           // 000000001DE0: BEBC0080
	s_cbranch_vccnz 65335                                      // 000000001DE4: BFA4FF37 <r_4_11_28_3_3_3+0x3c4>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001DE8: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DEC: BF870009
	s_add_u32 s20, s19, s20                                    // 000000001DF0: 80141413
	s_addc_u32 s21, s18, s21                                   // 000000001DF4: 82151512
	s_load_b32 s60, s[20:21], 0x4d8                            // 000000001DF8: F4000F0A F80004D8
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001E00: 8B6A077E
	s_cbranch_vccnz 65329                                      // 000000001E04: BFA4FF31 <r_4_11_28_3_3_3+0x3cc>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001E08: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E0C: BF870009
	s_add_u32 s20, s52, s20                                    // 000000001E10: 80141434
	s_addc_u32 s21, s51, s21                                   // 000000001E14: 82151533
	s_load_b32 s59, s[20:21], 0x4d0                            // 000000001E18: F4000ECA F80004D0
	s_mov_b32 s61, 0                                           // 000000001E20: BEBD0080
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001E24: 8B6A087E
	s_mov_b32 s62, 0                                           // 000000001E28: BEBE0080
	s_cbranch_vccnz 65323                                      // 000000001E2C: BFA4FF2B <r_4_11_28_3_3_3+0x3dc>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001E30: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E34: BF870009
	s_add_u32 s20, s52, s20                                    // 000000001E38: 80141434
	s_addc_u32 s21, s51, s21                                   // 000000001E3C: 82151533
	s_load_b32 s62, s[20:21], 0x4d4                            // 000000001E40: F4000F8A F80004D4
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001E48: 8B6A067E
	s_cbranch_vccnz 65317                                      // 000000001E4C: BFA4FF25 <r_4_11_28_3_3_3+0x3e4>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001E50: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E54: BF870009
	s_add_u32 s20, s52, s20                                    // 000000001E58: 80141434
	s_addc_u32 s21, s51, s21                                   // 000000001E5C: 82151533
	s_load_b32 s61, s[20:21], 0x4d8                            // 000000001E60: F4000F4A F80004D8
	s_mov_b32 s63, 0                                           // 000000001E68: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001E6C: 8B6A007E
	s_mov_b32 s64, 0                                           // 000000001E70: BEC00080
	s_cbranch_vccnz 65311                                      // 000000001E74: BFA4FF1F <r_4_11_28_3_3_3+0x3f4>
	s_lshl_b64 s[20:21], s[10:11], 2                           // 000000001E78: 8494820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E7C: BF870009
	s_add_u32 s20, s17, s20                                    // 000000001E80: 80141411
	s_addc_u32 s21, s16, s21                                   // 000000001E84: 82151510
	s_load_b32 s64, s[20:21], 0x9a0                            // 000000001E88: F400100A F80009A0
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001E90: 8B6A017E
	s_cbranch_vccnz 65305                                      // 000000001E94: BFA4FF19 <r_4_11_28_3_3_3+0x3fc>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001E98: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001E9C: BF870009
	s_add_u32 s0, s17, s0                                      // 000000001EA0: 80000011
	s_addc_u32 s1, s16, s1                                     // 000000001EA4: 82010110
	s_load_b32 s63, s[0:1], 0x9a4                              // 000000001EA8: F4000FC0 F80009A4
	s_mov_b32 s65, 0                                           // 000000001EB0: BEC10080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001EB4: 8B6A027E
	s_mov_b32 s2, 0                                            // 000000001EB8: BE820080
	s_cbranch_vccnz 65299                                      // 000000001EBC: BFA4FF13 <r_4_11_28_3_3_3+0x40c>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001EC0: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EC4: BF870009
	s_add_u32 s0, s17, s0                                      // 000000001EC8: 80000011
	s_addc_u32 s1, s16, s1                                     // 000000001ECC: 82010110
	s_load_b32 s2, s[0:1], 0x9a8                               // 000000001ED0: F4000080 F80009A8
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001ED8: 8B6A037E
	s_cbranch_vccnz 65293                                      // 000000001EDC: BFA4FF0D <r_4_11_28_3_3_3+0x414>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001EE0: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001EE4: BF870009
	s_add_u32 s0, s19, s0                                      // 000000001EE8: 80000013
	s_addc_u32 s1, s18, s1                                     // 000000001EEC: 82010112
	s_load_b32 s65, s[0:1], 0x9a0                              // 000000001EF0: F4001040 F80009A0
	s_mov_b32 s3, 0                                            // 000000001EF8: BE830080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001EFC: 8B6A047E
	s_mov_b32 s4, 0                                            // 000000001F00: BE840080
	s_cbranch_vccnz 65287                                      // 000000001F04: BFA4FF07 <r_4_11_28_3_3_3+0x424>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001F08: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F0C: BF870009
	s_add_u32 s0, s19, s0                                      // 000000001F10: 80000013
	s_addc_u32 s1, s18, s1                                     // 000000001F14: 82010112
	s_load_b32 s4, s[0:1], 0x9a4                               // 000000001F18: F4000100 F80009A4
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001F20: 8B6A057E
	s_cbranch_vccnz 65281                                      // 000000001F24: BFA4FF01 <r_4_11_28_3_3_3+0x42c>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001F28: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F2C: BF870009
	s_add_u32 s0, s19, s0                                      // 000000001F30: 80000013
	s_addc_u32 s1, s18, s1                                     // 000000001F34: 82010112
	s_load_b32 s3, s[0:1], 0x9a8                               // 000000001F38: F40000C0 F80009A8
	s_mov_b32 s5, 0                                            // 000000001F40: BE850080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001F44: 8B6A077E
	s_mov_b32 s7, 0                                            // 000000001F48: BE870080
	s_cbranch_vccnz 65275                                      // 000000001F4C: BFA4FEFB <r_4_11_28_3_3_3+0x43c>
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001F50: 8480820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F54: BF870009
	s_add_u32 s0, s52, s0                                      // 000000001F58: 80000034
	s_addc_u32 s1, s51, s1                                     // 000000001F5C: 82010133
	s_load_b32 s7, s[0:1], 0x9a0                               // 000000001F60: F40001C0 F80009A0
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000001F68: 8B6A087E
	s_cbranch_vccz 65269                                       // 000000001F6C: BFA3FEF5 <r_4_11_28_3_3_3+0x444>
	s_branch 65274                                             // 000000001F70: BFA0FEFA <r_4_11_28_3_3_3+0x45c>
