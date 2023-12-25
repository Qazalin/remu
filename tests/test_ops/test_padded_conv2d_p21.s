
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4_13_28_3_3_3>:
	s_load_b128 s[8:11], s[0:1], null                          // 000000001700: F4080200 F8000000
	s_mul_hi_i32 s2, s13, 0x92492493                           // 000000001708: 9702FF0D 92492493
	s_mul_i32 s6, s15, 0x39c                                   // 000000001710: 9606FF0F 0000039C
	s_add_i32 s2, s2, s13                                      // 000000001718: 81020D02
	s_mov_b32 s17, 0                                           // 00000000171C: BE910080
	s_lshr_b32 s3, s2, 31                                      // 000000001720: 85039F02
	s_ashr_i32 s2, s2, 4                                       // 000000001724: 86028402
	s_mov_b32 s37, s17                                         // 000000001728: BEA50011
	s_add_i32 s2, s2, s3                                       // 00000000172C: 81020302
	s_mov_b32 s33, 0                                           // 000000001730: BEA10080
	s_mul_i32 s4, s2, 28                                       // 000000001734: 96049C02
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000001738: F4040080 F8000010
	s_ashr_i32 s5, s4, 31                                      // 000000001740: 86059F04
	s_sub_i32 s36, s13, s4                                     // 000000001744: 81A4040D
	s_lshl_b64 s[34:35], s[4:5], 2                             // 000000001748: 84A28204
	s_waitcnt lgkmcnt(0)                                       // 00000000174C: BF89FC07
	s_add_u32 s0, s10, s34                                     // 000000001750: 8000220A
	s_addc_u32 s1, s11, s35                                    // 000000001754: 8201230B
	s_ashr_i32 s7, s6, 31                                      // 000000001758: 86079F06
	s_cmp_gt_i32 s36, 0                                        // 00000000175C: BF028024
	s_cselect_b32 s21, -1, 0                                   // 000000001760: 981580C1
	s_sub_i32 s4, s13, 56                                      // 000000001764: 8184B80D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001768: BF8704A9
	s_cmpk_lt_u32 s4, 0x134                                    // 00000000176C: B6840134
	s_cselect_b32 s4, -1, 0                                    // 000000001770: 980480C1
	s_and_b32 s5, s4, s21                                      // 000000001774: 8B051504
	s_add_u32 s23, s0, 0xffffff1c                              // 000000001778: 8017FF00 FFFFFF1C
	v_cndmask_b32_e64 v0, 0, 1, s5                             // 000000001780: D5010000 00150280
	s_addc_u32 s22, s1, -1                                     // 000000001788: 8216C101
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000178C: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870099
	s_add_u32 s18, s23, s6                                     // 000000001794: 80120617
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 000000001798: D44D0000 00020081
	s_addc_u32 s12, s22, s7                                    // 0000000017A0: 820C0716
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000017A4: 916A057E
	s_cbranch_vccnz 6                                          // 0000000017A8: BFA40006 <r_4_4_13_28_3_3_3+0xc4>
	s_lshl_b64 s[10:11], s[36:37], 2                           // 0000000017AC: 848A8224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B0: BF870009
	s_add_u32 s10, s18, s10                                    // 0000000017B4: 800A0A12
	s_addc_u32 s11, s12, s11                                   // 0000000017B8: 820B0B0C
	s_load_b32 s33, s[10:11], null                             // 0000000017BC: F4000845 F8000000
	s_mul_i32 s10, s14, 27                                     // 0000000017C4: 960A9B0E
	s_mov_b32 s40, s17                                         // 0000000017C8: BEA80011
	s_ashr_i32 s11, s10, 31                                    // 0000000017CC: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D0: BF870499
	s_lshl_b64 s[10:11], s[10:11], 2                           // 0000000017D4: 848A820A
	s_add_u32 s10, s2, s10                                     // 0000000017D8: 800A0A02
	s_addc_u32 s11, s3, s11                                    // 0000000017DC: 820B0B03
	s_add_i32 s16, s36, 1                                      // 0000000017E0: 81108124
	s_cmp_gt_i32 s36, -1                                       // 0000000017E4: BF02C124
	s_cselect_b32 s24, -1, 0                                   // 0000000017E8: 981880C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017EC: BF870499
	s_and_b32 s2, s4, s24                                      // 0000000017F0: 8B021804
	v_cndmask_b32_e64 v0, 0, 1, s2                             // 0000000017F4: D5010000 00090280
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000017FC: 916A027E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001800: BF870001
	v_cmp_ne_u32_e64 s1, 1, v0                                 // 000000001804: D44D0001 00020081
	s_cbranch_vccnz 6                                          // 00000000180C: BFA40006 <r_4_4_13_28_3_3_3+0x128>
	s_lshl_b64 s[2:3], s[16:17], 2                             // 000000001810: 84828210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001814: BF870009
	s_add_u32 s2, s18, s2                                      // 000000001818: 80020212
	s_addc_u32 s3, s12, s3                                     // 00000000181C: 8203030C
	s_load_b32 s40, s[2:3], null                               // 000000001820: F4000A01 F8000000
	s_cmp_lt_u32 s16, 28                                       // 000000001828: BF0A9C10
	s_mov_b32 s41, 0                                           // 00000000182C: BEA90080
	s_cselect_b32 s25, -1, 0                                   // 000000001830: 981980C1
	s_mov_b32 s42, 0                                           // 000000001834: BEAA0080
	s_and_b32 s3, s4, s25                                      // 000000001838: 8B031904
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000183C: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 000000001840: D5010000 000D0280
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001848: 916A037E
	v_cmp_ne_u32_e64 s2, 1, v0                                 // 00000000184C: D44D0002 00020081
	s_cbranch_vccnz 8                                          // 000000001854: BFA40008 <r_4_4_13_28_3_3_3+0x178>
	s_bfe_i64 s[4:5], s[36:37], 0x200000                       // 000000001858: 9484FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001860: BF870499
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001864: 84848204
	s_add_u32 s4, s18, s4                                      // 000000001868: 80040412
	s_addc_u32 s5, s12, s5                                     // 00000000186C: 8205050C
	s_load_b32 s42, s[4:5], 0x8                                // 000000001870: F4000A82 F8000008
	s_sub_i32 s3, s13, 28                                      // 000000001878: 81839C0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000187C: BF8704A9
	s_cmpk_lt_u32 s3, 0x134                                    // 000000001880: B6830134
	s_cselect_b32 s5, -1, 0                                    // 000000001884: 980580C1
	s_and_b32 s4, s5, s21                                      // 000000001888: 8B041505
	s_add_u32 s19, s23, s6                                     // 00000000188C: 80130617
	v_cndmask_b32_e64 v0, 0, 1, s4                             // 000000001890: D5010000 00110280
	s_addc_u32 s26, s22, s7                                    // 000000001898: 821A0716
	s_add_u32 s20, s19, 0x70                                   // 00000000189C: 8014FF13 00000070
	s_addc_u32 s19, s26, 0                                     // 0000000018A4: 8213801A
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 0000000018A8: 916A047E
	v_cmp_ne_u32_e64 s3, 1, v0                                 // 0000000018AC: D44D0003 00020081
	s_cbranch_vccnz 6                                          // 0000000018B4: BFA40006 <r_4_4_13_28_3_3_3+0x1d0>
	s_lshl_b64 s[26:27], s[36:37], 2                           // 0000000018B8: 849A8224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018BC: BF870009
	s_add_u32 s26, s20, s26                                    // 0000000018C0: 801A1A14
	s_addc_u32 s27, s19, s27                                   // 0000000018C4: 821B1B13
	s_load_b32 s41, s[26:27], null                             // 0000000018C8: F4000A4D F8000000
	s_and_b32 s26, s5, s24                                     // 0000000018D0: 8B1A1805
	s_mov_b32 s43, 0                                           // 0000000018D4: BEAB0080
	v_cndmask_b32_e64 v0, 0, 1, s26                            // 0000000018D8: D5010000 00690280
	s_and_not1_b32 vcc_lo, exec_lo, s26                        // 0000000018E0: 916A1A7E
	s_mov_b32 s44, 0                                           // 0000000018E4: BEAC0080
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018E8: BF870001
	v_cmp_ne_u32_e64 s4, 1, v0                                 // 0000000018EC: D44D0004 00020081
	s_cbranch_vccnz 6                                          // 0000000018F4: BFA40006 <r_4_4_13_28_3_3_3+0x210>
	s_lshl_b64 s[26:27], s[16:17], 2                           // 0000000018F8: 849A8210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018FC: BF870009
	s_add_u32 s26, s20, s26                                    // 000000001900: 801A1A14
	s_addc_u32 s27, s19, s27                                   // 000000001904: 821B1B13
	s_load_b32 s44, s[26:27], null                             // 000000001908: F4000B0D F8000000
	s_and_b32 s26, s5, s25                                     // 000000001910: 8B1A1905
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001914: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s26                            // 000000001918: D5010000 00690280
	s_and_not1_b32 vcc_lo, exec_lo, s26                        // 000000001920: 916A1A7E
	v_cmp_ne_u32_e64 s5, 1, v0                                 // 000000001924: D44D0005 00020081
	s_cbranch_vccnz 8                                          // 00000000192C: BFA40008 <r_4_4_13_28_3_3_3+0x250>
	s_bfe_i64 s[26:27], s[36:37], 0x200000                     // 000000001930: 949AFF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001938: BF870499
	s_lshl_b64 s[26:27], s[26:27], 2                           // 00000000193C: 849A821A
	s_add_u32 s26, s20, s26                                    // 000000001940: 801A1A14
	s_addc_u32 s27, s19, s27                                   // 000000001944: 821B1B13
	s_load_b32 s43, s[26:27], 0x8                              // 000000001948: F4000ACD F8000008
	s_add_i32 s13, s13, 27                                     // 000000001950: 810D9B0D
	s_mov_b32 s45, 0                                           // 000000001954: BEAD0080
	s_cmpk_lt_u32 s13, 0x14f                                   // 000000001958: B68D014F
	s_mov_b32 s46, 0                                           // 00000000195C: BEAE0080
	s_cselect_b32 s13, -1, 0                                   // 000000001960: 980D80C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001964: BF870009
	s_and_b32 s21, s13, s21                                    // 000000001968: 8B15150D
	s_add_u32 s23, s23, s6                                     // 00000000196C: 80170617
	v_cndmask_b32_e64 v0, 0, 1, s21                            // 000000001970: D5010000 00550280
	s_addc_u32 s7, s22, s7                                     // 000000001978: 82070716
	s_add_u32 s48, s23, 0xe0                                   // 00000000197C: 8030FF17 000000E0
	s_addc_u32 s47, s7, 0                                      // 000000001984: 822F8007
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001988: 916A157E
	v_cmp_ne_u32_e64 s6, 1, v0                                 // 00000000198C: D44D0006 00020081
	s_cbranch_vccnz 6                                          // 000000001994: BFA40006 <r_4_4_13_28_3_3_3+0x2b0>
	s_lshl_b64 s[22:23], s[36:37], 2                           // 000000001998: 84968224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000199C: BF870009
	s_add_u32 s22, s48, s22                                    // 0000000019A0: 80161630
	s_addc_u32 s23, s47, s23                                   // 0000000019A4: 8217172F
	s_load_b32 s46, s[22:23], null                             // 0000000019A8: F4000B8B F8000000
	s_and_b32 s21, s13, s24                                    // 0000000019B0: 8B15180D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019B4: BF8700A9
	v_cndmask_b32_e64 v0, 0, 1, s21                            // 0000000019B8: D5010000 00550280
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 0000000019C0: 916A157E
	v_cmp_ne_u32_e64 s7, 1, v0                                 // 0000000019C4: D44D0007 00020081
	s_cbranch_vccnz 6                                          // 0000000019CC: BFA40006 <r_4_4_13_28_3_3_3+0x2e8>
	s_lshl_b64 s[22:23], s[16:17], 2                           // 0000000019D0: 84968210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019D4: BF870009
	s_add_u32 s22, s48, s22                                    // 0000000019D8: 80161630
	s_addc_u32 s23, s47, s23                                   // 0000000019DC: 8217172F
	s_load_b32 s45, s[22:23], null                             // 0000000019E0: F4000B4B F8000000
	s_and_b32 s21, s13, s25                                    // 0000000019E8: 8B15190D
	s_mov_b32 s49, 0                                           // 0000000019EC: BEB10080
	s_xor_b32 s13, s21, -1                                     // 0000000019F0: 8D0DC115
	s_mov_b32 s50, 0                                           // 0000000019F4: BEB20080
	s_and_b32 vcc_lo, exec_lo, s13                             // 0000000019F8: 8B6A0D7E
	s_cbranch_vccz 57                                          // 0000000019FC: BFA30039 <r_4_4_13_28_3_3_3+0x3e4>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A00: 8B6A007E
	s_cbranch_vccz 65                                          // 000000001A04: BFA30041 <r_4_4_13_28_3_3_3+0x40c>
	s_mov_b32 s51, 0                                           // 000000001A08: BEB30080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A0C: 8B6A017E
	s_mov_b32 s52, 0                                           // 000000001A10: BEB40080
	s_cbranch_vccz 71                                          // 000000001A14: BFA30047 <r_4_4_13_28_3_3_3+0x434>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001A18: 8B6A027E
	s_cbranch_vccz 77                                          // 000000001A1C: BFA3004D <r_4_4_13_28_3_3_3+0x454>
	s_mov_b32 s53, 0                                           // 000000001A20: BEB50080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001A24: 8B6A037E
	s_mov_b32 s54, 0                                           // 000000001A28: BEB60080
	s_cbranch_vccz 85                                          // 000000001A2C: BFA30055 <r_4_4_13_28_3_3_3+0x484>
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001A30: 8B6A047E
	s_cbranch_vccz 91                                          // 000000001A34: BFA3005B <r_4_4_13_28_3_3_3+0x4a4>
	s_mov_b32 s55, 0                                           // 000000001A38: BEB70080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001A3C: 8B6A057E
	s_mov_b32 s56, 0                                           // 000000001A40: BEB80080
	s_cbranch_vccz 97                                          // 000000001A44: BFA30061 <r_4_4_13_28_3_3_3+0x4cc>
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001A48: 8B6A067E
	s_cbranch_vccz 105                                         // 000000001A4C: BFA30069 <r_4_4_13_28_3_3_3+0x4f4>
	s_mov_b32 s57, 0                                           // 000000001A50: BEB90080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001A54: 8B6A077E
	s_mov_b32 s58, 0                                           // 000000001A58: BEBA0080
	s_cbranch_vccz 111                                         // 000000001A5C: BFA3006F <r_4_4_13_28_3_3_3+0x51c>
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001A60: 916A157E
	s_cbranch_vccz 117                                         // 000000001A64: BFA30075 <r_4_4_13_28_3_3_3+0x53c>
	s_mov_b32 s59, 0                                           // 000000001A68: BEBB0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001A6C: 8B6A007E
	s_mov_b32 s60, 0                                           // 000000001A70: BEBC0080
	s_cbranch_vccz 125                                         // 000000001A74: BFA3007D <r_4_4_13_28_3_3_3+0x56c>
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001A78: 8B6A017E
	s_cbranch_vccz 131                                         // 000000001A7C: BFA30083 <r_4_4_13_28_3_3_3+0x58c>
	s_mov_b32 s61, 0                                           // 000000001A80: BEBD0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001A84: 8B6A027E
	s_mov_b32 s62, 0                                           // 000000001A88: BEBE0080
	s_cbranch_vccz 137                                         // 000000001A8C: BFA30089 <r_4_4_13_28_3_3_3+0x5b4>
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001A90: 8B6A037E
	s_cbranch_vccz 145                                         // 000000001A94: BFA30091 <r_4_4_13_28_3_3_3+0x5dc>
	s_mov_b32 s63, 0                                           // 000000001A98: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001A9C: 8B6A047E
	s_mov_b32 s64, 0                                           // 000000001AA0: BEC00080
	s_cbranch_vccz 151                                         // 000000001AA4: BFA30097 <r_4_4_13_28_3_3_3+0x604>
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001AA8: 8B6A057E
	s_cbranch_vccz 157                                         // 000000001AAC: BFA3009D <r_4_4_13_28_3_3_3+0x624>
	s_mov_b32 s65, 0                                           // 000000001AB0: BEC10080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001AB4: 8B6A067E
	s_mov_b32 s66, 0                                           // 000000001AB8: BEC20080
	s_cbranch_vccz 165                                         // 000000001ABC: BFA300A5 <r_4_4_13_28_3_3_3+0x654>
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001AC0: 8B6A077E
	s_cbranch_vccz 171                                         // 000000001AC4: BFA300AB <r_4_4_13_28_3_3_3+0x674>
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001AC8: 916A0D7E
	s_cbranch_vccnz 177                                        // 000000001ACC: BFA400B1 <r_4_4_13_28_3_3_3+0x694>
	s_bfe_i64 s[12:13], s[36:37], 0x200000                     // 000000001AD0: 948CFF24 00200000
	s_mov_b32 s68, 0                                           // 000000001AD8: BEC40080
	s_mov_b32 s67, 0                                           // 000000001ADC: BEC30080
	s_branch 173                                               // 000000001AE0: BFA000AD <r_4_4_13_28_3_3_3+0x698>
	s_bfe_i64 s[22:23], s[36:37], 0x200000                     // 000000001AE4: 9496FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001AEC: BF870499
	s_lshl_b64 s[22:23], s[22:23], 2                           // 000000001AF0: 84968216
	s_add_u32 s22, s48, s22                                    // 000000001AF4: 80161630
	s_addc_u32 s23, s47, s23                                   // 000000001AF8: 8217172F
	s_load_b32 s50, s[22:23], 0x8                              // 000000001AFC: F4000C8B F8000008
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001B04: 8B6A007E
	s_cbranch_vccnz 65471                                      // 000000001B08: BFA4FFBF <r_4_4_13_28_3_3_3+0x308>
	s_lshl_b64 s[22:23], s[36:37], 2                           // 000000001B0C: 84968224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B10: BF870009
	s_add_u32 s22, s18, s22                                    // 000000001B14: 80161612
	s_addc_u32 s23, s12, s23                                   // 000000001B18: 8217170C
	s_load_b32 s49, s[22:23], 0x4d0                            // 000000001B1C: F4000C4B F80004D0
	s_mov_b32 s51, 0                                           // 000000001B24: BEB30080
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001B28: 8B6A017E
	s_mov_b32 s52, 0                                           // 000000001B2C: BEB40080
	s_cbranch_vccnz 65465                                      // 000000001B30: BFA4FFB9 <r_4_4_13_28_3_3_3+0x318>
	s_lshl_b64 s[22:23], s[16:17], 2                           // 000000001B34: 84968210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B38: BF870009
	s_add_u32 s22, s18, s22                                    // 000000001B3C: 80161612
	s_addc_u32 s23, s12, s23                                   // 000000001B40: 8217170C
	s_load_b32 s52, s[22:23], 0x4d0                            // 000000001B44: F4000D0B F80004D0
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001B4C: 8B6A027E
	s_cbranch_vccnz 65459                                      // 000000001B50: BFA4FFB3 <r_4_4_13_28_3_3_3+0x320>
	s_bfe_i64 s[22:23], s[36:37], 0x200000                     // 000000001B54: 9496FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001B5C: BF870499
	s_lshl_b64 s[22:23], s[22:23], 2                           // 000000001B60: 84968216
	s_add_u32 s22, s18, s22                                    // 000000001B64: 80161612
	s_addc_u32 s23, s12, s23                                   // 000000001B68: 8217170C
	s_load_b32 s51, s[22:23], 0x4d8                            // 000000001B6C: F4000CCB F80004D8
	s_mov_b32 s53, 0                                           // 000000001B74: BEB50080
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001B78: 8B6A037E
	s_mov_b32 s54, 0                                           // 000000001B7C: BEB60080
	s_cbranch_vccnz 65451                                      // 000000001B80: BFA4FFAB <r_4_4_13_28_3_3_3+0x330>
	s_lshl_b64 s[22:23], s[36:37], 2                           // 000000001B84: 84968224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B88: BF870009
	s_add_u32 s22, s20, s22                                    // 000000001B8C: 80161614
	s_addc_u32 s23, s19, s23                                   // 000000001B90: 82171713
	s_load_b32 s54, s[22:23], 0x4d0                            // 000000001B94: F4000D8B F80004D0
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001B9C: 8B6A047E
	s_cbranch_vccnz 65445                                      // 000000001BA0: BFA4FFA5 <r_4_4_13_28_3_3_3+0x338>
	s_lshl_b64 s[22:23], s[16:17], 2                           // 000000001BA4: 84968210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001BA8: BF870009
	s_add_u32 s22, s20, s22                                    // 000000001BAC: 80161614
	s_addc_u32 s23, s19, s23                                   // 000000001BB0: 82171713
	s_load_b32 s53, s[22:23], 0x4d0                            // 000000001BB4: F4000D4B F80004D0
	s_mov_b32 s55, 0                                           // 000000001BBC: BEB70080
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001BC0: 8B6A057E
	s_mov_b32 s56, 0                                           // 000000001BC4: BEB80080
	s_cbranch_vccnz 65439                                      // 000000001BC8: BFA4FF9F <r_4_4_13_28_3_3_3+0x348>
	s_bfe_i64 s[22:23], s[36:37], 0x200000                     // 000000001BCC: 9496FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001BD4: BF870499
	s_lshl_b64 s[22:23], s[22:23], 2                           // 000000001BD8: 84968216
	s_add_u32 s22, s20, s22                                    // 000000001BDC: 80161614
	s_addc_u32 s23, s19, s23                                   // 000000001BE0: 82171713
	s_load_b32 s56, s[22:23], 0x4d8                            // 000000001BE4: F4000E0B F80004D8
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001BEC: 8B6A067E
	s_cbranch_vccnz 65431                                      // 000000001BF0: BFA4FF97 <r_4_4_13_28_3_3_3+0x350>
	s_lshl_b64 s[22:23], s[36:37], 2                           // 000000001BF4: 84968224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001BF8: BF870009
	s_add_u32 s22, s48, s22                                    // 000000001BFC: 80161630
	s_addc_u32 s23, s47, s23                                   // 000000001C00: 8217172F
	s_load_b32 s55, s[22:23], 0x4d0                            // 000000001C04: F4000DCB F80004D0
	s_mov_b32 s57, 0                                           // 000000001C0C: BEB90080
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001C10: 8B6A077E
	s_mov_b32 s58, 0                                           // 000000001C14: BEBA0080
	s_cbranch_vccnz 65425                                      // 000000001C18: BFA4FF91 <r_4_4_13_28_3_3_3+0x360>
	s_lshl_b64 s[22:23], s[16:17], 2                           // 000000001C1C: 84968210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C20: BF870009
	s_add_u32 s22, s48, s22                                    // 000000001C24: 80161630
	s_addc_u32 s23, s47, s23                                   // 000000001C28: 8217172F
	s_load_b32 s58, s[22:23], 0x4d0                            // 000000001C2C: F4000E8B F80004D0
	s_and_not1_b32 vcc_lo, exec_lo, s21                        // 000000001C34: 916A157E
	s_cbranch_vccnz 65419                                      // 000000001C38: BFA4FF8B <r_4_4_13_28_3_3_3+0x368>
	s_bfe_i64 s[22:23], s[36:37], 0x200000                     // 000000001C3C: 9496FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C44: BF870499
	s_lshl_b64 s[22:23], s[22:23], 2                           // 000000001C48: 84968216
	s_add_u32 s22, s48, s22                                    // 000000001C4C: 80161630
	s_addc_u32 s23, s47, s23                                   // 000000001C50: 8217172F
	s_load_b32 s57, s[22:23], 0x4d8                            // 000000001C54: F4000E4B F80004D8
	s_mov_b32 s59, 0                                           // 000000001C5C: BEBB0080
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001C60: 8B6A007E
	s_mov_b32 s60, 0                                           // 000000001C64: BEBC0080
	s_cbranch_vccnz 65411                                      // 000000001C68: BFA4FF83 <r_4_4_13_28_3_3_3+0x378>
	s_lshl_b64 s[22:23], s[36:37], 2                           // 000000001C6C: 84968224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C70: BF870009
	s_add_u32 s22, s18, s22                                    // 000000001C74: 80161612
	s_addc_u32 s23, s12, s23                                   // 000000001C78: 8217170C
	s_load_b32 s60, s[22:23], 0x9a0                            // 000000001C7C: F4000F0B F80009A0
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001C84: 8B6A017E
	s_cbranch_vccnz 65405                                      // 000000001C88: BFA4FF7D <r_4_4_13_28_3_3_3+0x380>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 000000001C8C: 84808210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C90: BF870009
	s_add_u32 s0, s18, s0                                      // 000000001C94: 80000012
	s_addc_u32 s1, s12, s1                                     // 000000001C98: 8201010C
	s_load_b32 s59, s[0:1], 0x9a0                              // 000000001C9C: F4000EC0 F80009A0
	s_mov_b32 s61, 0                                           // 000000001CA4: BEBD0080
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001CA8: 8B6A027E
	s_mov_b32 s62, 0                                           // 000000001CAC: BEBE0080
	s_cbranch_vccnz 65399                                      // 000000001CB0: BFA4FF77 <r_4_4_13_28_3_3_3+0x390>
	s_bfe_i64 s[0:1], s[36:37], 0x200000                       // 000000001CB4: 9480FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001CBC: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001CC0: 84808200
	s_add_u32 s0, s18, s0                                      // 000000001CC4: 80000012
	s_addc_u32 s1, s12, s1                                     // 000000001CC8: 8201010C
	s_load_b32 s62, s[0:1], 0x9a8                              // 000000001CCC: F4000F80 F80009A8
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000001CD4: 8B6A037E
	s_cbranch_vccnz 65391                                      // 000000001CD8: BFA4FF6F <r_4_4_13_28_3_3_3+0x398>
	s_lshl_b64 s[0:1], s[36:37], 2                             // 000000001CDC: 84808224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CE0: BF870009
	s_add_u32 s0, s20, s0                                      // 000000001CE4: 80000014
	s_addc_u32 s1, s19, s1                                     // 000000001CE8: 82010113
	s_load_b32 s61, s[0:1], 0x9a0                              // 000000001CEC: F4000F40 F80009A0
	s_mov_b32 s63, 0                                           // 000000001CF4: BEBF0080
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000001CF8: 8B6A047E
	s_mov_b32 s64, 0                                           // 000000001CFC: BEC00080
	s_cbranch_vccnz 65385                                      // 000000001D00: BFA4FF69 <r_4_4_13_28_3_3_3+0x3a8>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 000000001D04: 84808210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D08: BF870009
	s_add_u32 s0, s20, s0                                      // 000000001D0C: 80000014
	s_addc_u32 s1, s19, s1                                     // 000000001D10: 82010113
	s_load_b32 s64, s[0:1], 0x9a0                              // 000000001D14: F4001000 F80009A0
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000001D1C: 8B6A057E
	s_cbranch_vccnz 65379                                      // 000000001D20: BFA4FF63 <r_4_4_13_28_3_3_3+0x3b0>
	s_bfe_i64 s[0:1], s[36:37], 0x200000                       // 000000001D24: 9480FF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001D2C: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001D30: 84808200
	s_add_u32 s0, s20, s0                                      // 000000001D34: 80000014
	s_addc_u32 s1, s19, s1                                     // 000000001D38: 82010113
	s_load_b32 s63, s[0:1], 0x9a8                              // 000000001D3C: F4000FC0 F80009A8
	s_mov_b32 s65, 0                                           // 000000001D44: BEC10080
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001D48: 8B6A067E
	s_mov_b32 s66, 0                                           // 000000001D4C: BEC20080
	s_cbranch_vccnz 65371                                      // 000000001D50: BFA4FF5B <r_4_4_13_28_3_3_3+0x3c0>
	s_lshl_b64 s[0:1], s[36:37], 2                             // 000000001D54: 84808224
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D58: BF870009
	s_add_u32 s0, s48, s0                                      // 000000001D5C: 80000030
	s_addc_u32 s1, s47, s1                                     // 000000001D60: 8201012F
	s_load_b32 s66, s[0:1], 0x9a0                              // 000000001D64: F4001080 F80009A0
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000001D6C: 8B6A077E
	s_cbranch_vccnz 65365                                      // 000000001D70: BFA4FF55 <r_4_4_13_28_3_3_3+0x3c8>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 000000001D74: 84808210
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D78: BF870009
	s_add_u32 s0, s48, s0                                      // 000000001D7C: 80000030
	s_addc_u32 s1, s47, s1                                     // 000000001D80: 8201012F
	s_load_b32 s65, s[0:1], 0x9a0                              // 000000001D84: F4001040 F80009A0
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000001D8C: 916A0D7E
	s_cbranch_vccz 65359                                       // 000000001D90: BFA3FF4F <r_4_4_13_28_3_3_3+0x3d0>
	s_mov_b32 s68, -1                                          // 000000001D94: BEC400C1
	s_clause 0x3                                               // 000000001D98: BF850003
	s_load_b256 s[24:31], s[10:11], null                       // 000000001D9C: F40C0605 F8000000
	s_load_b256 s[16:23], s[10:11], 0x20                       // 000000001DA4: F40C0405 F8000020
	s_load_b256 s[0:7], s[10:11], 0x40                         // 000000001DAC: F40C0005 F8000040
	s_load_b64 s[38:39], s[10:11], 0x60                        // 000000001DB4: F4040985 F8000060
	s_and_not1_b32 vcc_lo, exec_lo, s68                        // 000000001DBC: 916A447E
	s_cbranch_vccnz 8                                          // 000000001DC0: BFA40008 <r_4_4_13_28_3_3_3+0x6e4>
	s_bfe_i64 s[12:13], s[36:37], 0x200000                     // 000000001DC4: 948CFF24 00200000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001DCC: BF870499
	s_lshl_b64 s[36:37], s[12:13], 2                           // 000000001DD0: 84A4820C
	s_add_u32 s36, s48, s36                                    // 000000001DD4: 80242430
	s_addc_u32 s37, s47, s37                                   // 000000001DD8: 8225252F
	s_load_b32 s67, s[36:37], 0x9a8                            // 000000001DDC: F40010D2 F80009A8
	s_waitcnt lgkmcnt(0)                                       // 000000001DE4: BF89FC07
	v_fma_f32 v0, s33, s24, 0                                  // 000000001DE8: D6130000 02003021
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001DF0: BF870091
	v_fmac_f32_e64 v0, s40, s25                                // 000000001DF4: D52B0000 00003228
	v_fmac_f32_e64 v0, s42, s26                                // 000000001DFC: D52B0000 0000342A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E04: BF870091
	v_fmac_f32_e64 v0, s41, s27                                // 000000001E08: D52B0000 00003629
	v_fmac_f32_e64 v0, s44, s28                                // 000000001E10: D52B0000 0000382C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E18: BF870091
	v_fmac_f32_e64 v0, s43, s29                                // 000000001E1C: D52B0000 00003A2B
	v_fmac_f32_e64 v0, s46, s30                                // 000000001E24: D52B0000 00003C2E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E2C: BF870091
	v_fmac_f32_e64 v0, s45, s31                                // 000000001E30: D52B0000 00003E2D
	v_fmac_f32_e64 v0, s50, s16                                // 000000001E38: D52B0000 00002032
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E40: BF870091
	v_fmac_f32_e64 v0, s49, s17                                // 000000001E44: D52B0000 00002231
	v_fmac_f32_e64 v0, s52, s18                                // 000000001E4C: D52B0000 00002434
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E54: BF870091
	v_fmac_f32_e64 v0, s51, s19                                // 000000001E58: D52B0000 00002633
	v_fmac_f32_e64 v0, s54, s20                                // 000000001E60: D52B0000 00002836
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E68: BF870091
	v_fmac_f32_e64 v0, s53, s21                                // 000000001E6C: D52B0000 00002A35
	v_fmac_f32_e64 v0, s56, s22                                // 000000001E74: D52B0000 00002C38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E7C: BF870091
	v_fmac_f32_e64 v0, s55, s23                                // 000000001E80: D52B0000 00002E37
	v_fmac_f32_e64 v0, s58, s0                                 // 000000001E88: D52B0000 0000003A
	s_mul_i32 s0, s15, 0x5b0                                   // 000000001E90: 9600FF0F 000005B0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001E98: BF8704A1
	v_fmac_f32_e64 v0, s57, s1                                 // 000000001E9C: D52B0000 00000239
	s_ashr_i32 s1, s0, 31                                      // 000000001EA4: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001EA8: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001EAC: BF8700A1
	v_fmac_f32_e64 v0, s60, s2                                 // 000000001EB0: D52B0000 0000043C
	s_mul_i32 s2, s14, 0x16c                                   // 000000001EB8: 9602FF0E 0000016C
	v_fmac_f32_e64 v0, s59, s3                                 // 000000001EC0: D52B0000 0000063B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001EC8: BF8700A1
	v_fmac_f32_e64 v0, s62, s4                                 // 000000001ECC: D52B0000 0000083E
	s_load_b32 s4, s[10:11], 0x68                              // 000000001ED4: F4000105 F8000068
	v_fmac_f32_e64 v0, s61, s5                                 // 000000001EDC: D52B0000 00000A3D
	s_add_u32 s5, s8, s0                                       // 000000001EE4: 80050008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001EE8: BF8704B1
	v_fmac_f32_e64 v0, s64, s6                                 // 000000001EEC: D52B0000 00000C40
	s_addc_u32 s6, s9, s1                                      // 000000001EF4: 82060109
	s_ashr_i32 s3, s2, 31                                      // 000000001EF8: 86039F02
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001EFC: 84808202
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001F00: BF870001
	v_fmac_f32_e64 v0, s63, s7                                 // 000000001F04: D52B0000 00000E3F
	s_add_u32 s0, s5, s0                                       // 000000001F0C: 80000005
	s_addc_u32 s1, s6, s1                                      // 000000001F10: 82010106
	s_add_u32 s2, s0, s34                                      // 000000001F14: 80022200
	s_addc_u32 s3, s1, s35                                     // 000000001F18: 82032301
	v_fmac_f32_e64 v0, s66, s38                                // 000000001F1C: D52B0000 00004C42
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001F24: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001F28: BF8700A9
	s_add_u32 s0, s2, s0                                       // 000000001F2C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001F30: 82010103
	v_fmac_f32_e64 v0, s65, s39                                // 000000001F34: D52B0000 00004E41
	s_waitcnt lgkmcnt(0)                                       // 000000001F3C: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001F40: BF870091
	v_fmac_f32_e64 v0, s67, s4                                 // 000000001F44: D52B0000 00000843
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001F4C: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001F54: DC6A0000 00000001
	s_nop 0                                                    // 000000001F5C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001F60: BFB60003
	s_endpgm                                                   // 000000001F64: BFB00000
