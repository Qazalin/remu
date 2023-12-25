
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_6_9_2_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[8:11], s[0:1], null                          // 000000001704: F4080200 F8000000
	s_load_b64 s[2:3], s[0:1], 0x10                            // 00000000170C: F4040080 F8000010
	s_mul_i32 s16, s15, 22                                     // 000000001714: 9610960F
	s_mov_b32 s12, s13                                         // 000000001718: BE8C000D
	s_mov_b32 s20, 0                                           // 00000000171C: BE940080
	s_mov_b32 s21, 0                                           // 000000001720: BE950080
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s19, s10, -4                                     // 000000001728: 8013C40A
	s_addc_u32 s18, s11, -1                                    // 00000000172C: 8212C10B
	s_add_i32 s0, s13, -1                                      // 000000001730: 8100C10D
	s_ashr_i32 s17, s16, 31                                    // 000000001734: 86119F10
	s_cmp_gt_i32 s0, -1                                        // 000000001738: BF02C100
	s_cselect_b32 s0, -1, 0                                    // 00000000173C: 980080C1
	s_cmp_lt_i32 s13, 12                                       // 000000001740: BF048C0D
	s_cselect_b32 s1, -1, 0                                    // 000000001744: 980180C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001748: BF870499
	s_and_b32 s1, s1, s0                                       // 00000000174C: 8B010001
	v_cndmask_b32_e64 v0, 0, 1, s1                             // 000000001750: D5010000 00050280
	s_and_not1_b32 vcc_lo, exec_lo, s1                         // 000000001758: 916A017E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000175C: BF870001
	v_cmp_ne_u32_e64 s0, 1, v0                                 // 000000001760: D44D0000 00020081
	s_cbranch_vccnz 10                                         // 000000001768: BFA4000A <r_6_6_9_2_5+0x94>
	s_lshl_b64 s[4:5], s[16:17], 2                             // 00000000176C: 84848210
	s_ashr_i32 s13, s12, 31                                    // 000000001770: 860D9F0C
	s_add_u32 s1, s19, s4                                      // 000000001774: 80010413
	s_addc_u32 s6, s18, s5                                     // 000000001778: 82060512
	s_lshl_b64 s[4:5], s[12:13], 2                             // 00000000177C: 8484820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s4, s1, s4                                       // 000000001784: 80040401
	s_addc_u32 s5, s6, s5                                      // 000000001788: 82050506
	s_load_b32 s21, s[4:5], null                               // 00000000178C: F4000542 F8000000
	s_mul_i32 s4, s14, 10                                      // 000000001794: 96048A0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_ashr_i32 s5, s4, 31                                      // 00000000179C: 86059F04
	s_lshl_b64 s[4:5], s[4:5], 2                               // 0000000017A0: 84848204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A4: BF870009
	s_add_u32 s10, s2, s4                                      // 0000000017A8: 800A0402
	s_addc_u32 s11, s3, s5                                     // 0000000017AC: 820B0503
	s_add_i32 s2, s12, 1                                       // 0000000017B0: 8102810C
	s_cmp_lt_u32 s12, 11                                       // 0000000017B4: BF0A8B0C
	s_cselect_b32 s1, -1, 0                                    // 0000000017B8: 980180C1
	s_cmp_gt_u32 s12, 10                                       // 0000000017BC: BF088A0C
	s_cbranch_scc1 10                                          // 0000000017C0: BFA2000A <r_6_6_9_2_5+0xec>
	s_lshl_b64 s[4:5], s[16:17], 2                             // 0000000017C4: 84848210
	s_mov_b32 s3, 0                                            // 0000000017C8: BE830080
	s_add_u32 s6, s19, s4                                      // 0000000017CC: 80060413
	s_addc_u32 s7, s18, s5                                     // 0000000017D0: 82070512
	s_lshl_b64 s[4:5], s[2:3], 2                               // 0000000017D4: 84848202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017D8: BF870009
	s_add_u32 s4, s6, s4                                       // 0000000017DC: 80040406
	s_addc_u32 s5, s7, s5                                      // 0000000017E0: 82050507
	s_load_b32 s20, s[4:5], null                               // 0000000017E4: F4000502 F8000000
	s_cmp_lt_u32 s2, 11                                        // 0000000017EC: BF0A8B02
	s_mov_b32 s22, 0                                           // 0000000017F0: BE960080
	s_cselect_b32 s5, -1, 0                                    // 0000000017F4: 980580C1
	s_cmp_gt_u32 s2, 10                                        // 0000000017F8: BF088A02
	s_mov_b32 s23, 0                                           // 0000000017FC: BE970080
	s_cbranch_scc1 11                                          // 000000001800: BFA2000B <r_6_6_9_2_5+0x130>
	s_lshl_b64 s[6:7], s[16:17], 2                             // 000000001804: 84868210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001808: BF8704B9
	s_add_u32 s3, s19, s6                                      // 00000000180C: 80030613
	s_addc_u32 s4, s18, s7                                     // 000000001810: 82040712
	s_ashr_i32 s13, s12, 31                                    // 000000001814: 860D9F0C
	s_lshl_b64 s[6:7], s[12:13], 2                             // 000000001818: 8486820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000181C: BF870009
	s_add_u32 s6, s3, s6                                       // 000000001820: 80060603
	s_addc_u32 s7, s4, s7                                      // 000000001824: 82070704
	s_load_b32 s23, s[6:7], 0x8                                // 000000001828: F40005C3 F8000008
	s_add_i32 s3, s12, 2                                       // 000000001830: 8103820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001834: BF870009
	s_cmp_lt_u32 s3, 11                                        // 000000001838: BF0A8B03
	s_cselect_b32 s6, -1, 0                                    // 00000000183C: 980680C1
	s_cmp_gt_u32 s3, 10                                        // 000000001840: BF088A03
	s_cbranch_scc1 11                                          // 000000001844: BFA2000B <r_6_6_9_2_5+0x174>
	s_lshl_b64 s[24:25], s[16:17], 2                           // 000000001848: 84988210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000184C: BF8704B9
	s_add_u32 s3, s19, s24                                     // 000000001850: 80031813
	s_addc_u32 s4, s18, s25                                    // 000000001854: 82041912
	s_ashr_i32 s13, s12, 31                                    // 000000001858: 860D9F0C
	s_lshl_b64 s[24:25], s[12:13], 2                           // 00000000185C: 8498820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001860: BF870009
	s_add_u32 s24, s3, s24                                     // 000000001864: 80181803
	s_addc_u32 s25, s4, s25                                    // 000000001868: 82191904
	s_load_b32 s22, s[24:25], 0xc                              // 00000000186C: F400058C F800000C
	s_add_i32 s3, s12, 3                                       // 000000001874: 8103830C
	s_mov_b32 s24, 0                                           // 000000001878: BE980080
	s_cmp_gt_u32 s3, 10                                        // 00000000187C: BF088A03
	s_mov_b32 s25, 0                                           // 000000001880: BE990080
	s_cselect_b32 s4, -1, 0                                    // 000000001884: 980480C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001888: BF870009
	s_and_b32 vcc_lo, exec_lo, s4                              // 00000000188C: 8B6A047E
	s_cbranch_vccz 22                                          // 000000001890: BFA30016 <r_6_6_9_2_5+0x1ec>
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001894: 8B6A007E
	s_cbranch_vccz 33                                          // 000000001898: BFA30021 <r_6_6_9_2_5+0x220>
	s_mov_b32 s26, 0                                           // 00000000189C: BE9A0080
	s_and_not1_b32 vcc_lo, exec_lo, s1                         // 0000000018A0: 916A017E
	s_mov_b32 s27, 0                                           // 0000000018A4: BE9B0080
	s_cbranch_vccz 43                                          // 0000000018A8: BFA3002B <r_6_6_9_2_5+0x258>
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 0000000018AC: 916A057E
	s_cbranch_vccz 53                                          // 0000000018B0: BFA30035 <r_6_6_9_2_5+0x288>
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 0000000018B4: 916A067E
	s_mov_b32 s28, 0                                           // 0000000018B8: BE9C0080
	s_cbranch_vccz 64                                          // 0000000018BC: BFA30040 <r_6_6_9_2_5+0x2c0>
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 0000000018C0: 916A047E
	s_cbranch_vccnz 75                                         // 0000000018C4: BFA4004B <r_6_6_9_2_5+0x2f4>
	s_ashr_i32 s13, s12, 31                                    // 0000000018C8: 860D9F0C
	s_mov_b32 s29, 0                                           // 0000000018CC: BE9D0080
	s_clause 0x1                                               // 0000000018D0: BF850001
	s_load_b256 s[0:7], s[10:11], null                         // 0000000018D4: F40C0005 F8000000
	s_load_b32 s30, s[10:11], 0x20                             // 0000000018DC: F4000785 F8000020
	s_cbranch_execz 72                                         // 0000000018E4: BFA50048 <r_6_6_9_2_5+0x308>
	s_branch 82                                                // 0000000018E8: BFA00052 <r_6_6_9_2_5+0x334>
	s_lshl_b64 s[26:27], s[16:17], 2                           // 0000000018EC: 849A8210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018F0: BF8704B9
	s_add_u32 s3, s19, s26                                     // 0000000018F4: 80031A13
	s_addc_u32 s7, s18, s27                                    // 0000000018F8: 82071B12
	s_ashr_i32 s13, s12, 31                                    // 0000000018FC: 860D9F0C
	s_lshl_b64 s[26:27], s[12:13], 2                           // 000000001900: 849A820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001904: BF870009
	s_add_u32 s26, s3, s26                                     // 000000001908: 801A1A03
	s_addc_u32 s27, s7, s27                                    // 00000000190C: 821B1B07
	s_load_b32 s25, s[26:27], 0x10                             // 000000001910: F400064D F8000010
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001918: 8B6A007E
	s_cbranch_vccnz 65503                                      // 00000000191C: BFA4FFDF <r_6_6_9_2_5+0x19c>
	s_lshl_b64 s[26:27], s[16:17], 2                           // 000000001920: 849A8210
	s_ashr_i32 s13, s12, 31                                    // 000000001924: 860D9F0C
	s_add_u32 s0, s19, s26                                     // 000000001928: 80001A13
	s_addc_u32 s3, s18, s27                                    // 00000000192C: 82031B12
	s_lshl_b64 s[26:27], s[12:13], 2                           // 000000001930: 849A820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001934: BF870009
	s_add_u32 s26, s0, s26                                     // 000000001938: 801A1A00
	s_addc_u32 s27, s3, s27                                    // 00000000193C: 821B1B03
	s_load_b32 s24, s[26:27], 0x2c                             // 000000001940: F400060D F800002C
	s_mov_b32 s26, 0                                           // 000000001948: BE9A0080
	s_and_not1_b32 vcc_lo, exec_lo, s1                         // 00000000194C: 916A017E
	s_mov_b32 s27, 0                                           // 000000001950: BE9B0080
	s_cbranch_vccnz 65493                                      // 000000001954: BFA4FFD5 <r_6_6_9_2_5+0x1ac>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 000000001958: 84808210
	s_mov_b32 s3, 0                                            // 00000000195C: BE830080
	s_add_u32 s7, s19, s0                                      // 000000001960: 80070013
	s_addc_u32 s13, s18, s1                                    // 000000001964: 820D0112
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001968: 84808202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000196C: BF870009
	s_add_u32 s0, s7, s0                                       // 000000001970: 80000007
	s_addc_u32 s1, s13, s1                                     // 000000001974: 8201010D
	s_load_b32 s27, s[0:1], 0x2c                               // 000000001978: F40006C0 F800002C
	s_and_not1_b32 vcc_lo, exec_lo, s5                         // 000000001980: 916A057E
	s_cbranch_vccnz 65483                                      // 000000001984: BFA4FFCB <r_6_6_9_2_5+0x1b4>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 000000001988: 84808210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000198C: BF8704B9
	s_add_u32 s2, s19, s0                                      // 000000001990: 80020013
	s_addc_u32 s3, s18, s1                                     // 000000001994: 82030112
	s_ashr_i32 s13, s12, 31                                    // 000000001998: 860D9F0C
	s_lshl_b64 s[0:1], s[12:13], 2                             // 00000000199C: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019A0: BF870009
	s_add_u32 s0, s2, s0                                       // 0000000019A4: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000019A8: 82010103
	s_load_b32 s26, s[0:1], 0x34                               // 0000000019AC: F4000680 F8000034
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 0000000019B4: 916A067E
	s_mov_b32 s28, 0                                           // 0000000019B8: BE9C0080
	s_cbranch_vccnz 65472                                      // 0000000019BC: BFA4FFC0 <r_6_6_9_2_5+0x1c0>
	s_lshl_b64 s[0:1], s[16:17], 2                             // 0000000019C0: 84808210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000019C4: BF8704B9
	s_add_u32 s2, s19, s0                                      // 0000000019C8: 80020013
	s_addc_u32 s3, s18, s1                                     // 0000000019CC: 82030112
	s_ashr_i32 s13, s12, 31                                    // 0000000019D0: 860D9F0C
	s_lshl_b64 s[0:1], s[12:13], 2                             // 0000000019D4: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000019D8: BF870009
	s_add_u32 s0, s2, s0                                       // 0000000019DC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000019E0: 82010103
	s_load_b32 s28, s[0:1], 0x38                               // 0000000019E4: F4000700 F8000038
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 0000000019EC: 916A047E
	s_cbranch_vccz 65461                                       // 0000000019F0: BFA3FFB5 <r_6_6_9_2_5+0x1c8>
	s_clause 0x1                                               // 0000000019F4: BF850001
	s_load_b256 s[0:7], s[10:11], null                         // 0000000019F8: F40C0005 F8000000
	s_load_b32 s30, s[10:11], 0x20                             // 000000001A00: F4000785 F8000020
	s_lshl_b64 s[16:17], s[16:17], 2                           // 000000001A08: 84908210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001A0C: BF8704B9
	s_add_u32 s19, s19, s16                                    // 000000001A10: 80131013
	s_addc_u32 s18, s18, s17                                   // 000000001A14: 82121112
	s_ashr_i32 s13, s12, 31                                    // 000000001A18: 860D9F0C
	s_lshl_b64 s[16:17], s[12:13], 2                           // 000000001A1C: 8490820C
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001A20: BF870009
	s_add_u32 s16, s19, s16                                    // 000000001A24: 80101013
	s_addc_u32 s17, s18, s17                                   // 000000001A28: 82111112
	s_load_b32 s29, s[16:17], 0x3c                             // 000000001A2C: F4000748 F800003C
	s_waitcnt lgkmcnt(0)                                       // 000000001A34: BF89FC07
	v_fma_f32 v0, s21, s0, 0                                   // 000000001A38: D6130000 02000015
	s_mul_i32 s0, s15, 54                                      // 000000001A40: 9600B60F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001A44: BF8704A1
	v_fmac_f32_e64 v0, s20, s1                                 // 000000001A48: D52B0000 00000214
	s_ashr_i32 s1, s0, 31                                      // 000000001A50: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001A54: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A58: BF8700A1
	v_fmac_f32_e64 v0, s23, s2                                 // 000000001A5C: D52B0000 00000417
	s_mul_i32 s2, s14, 9                                       // 000000001A64: 9602890E
	v_fmac_f32_e64 v0, s22, s3                                 // 000000001A68: D52B0000 00000616
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001A70: BF8700A1
	v_fmac_f32_e64 v0, s25, s4                                 // 000000001A74: D52B0000 00000819
	s_load_b32 s4, s[10:11], 0x24                              // 000000001A7C: F4000105 F8000024
	v_fmac_f32_e64 v0, s24, s5                                 // 000000001A84: D52B0000 00000A18
	s_add_u32 s5, s8, s0                                       // 000000001A8C: 80050008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001A90: BF8704B1
	v_fmac_f32_e64 v0, s27, s6                                 // 000000001A94: D52B0000 00000C1B
	s_addc_u32 s6, s9, s1                                      // 000000001A9C: 82060109
	s_ashr_i32 s3, s2, 31                                      // 000000001AA0: 86039F02
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001AA4: 84808202
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001AA8: BF8704C1
	v_fmac_f32_e64 v0, s26, s7                                 // 000000001AAC: D52B0000 00000E1A
	s_add_u32 s2, s5, s0                                       // 000000001AB4: 80020005
	s_addc_u32 s3, s6, s1                                      // 000000001AB8: 82030106
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001ABC: 8480820C
	s_add_u32 s0, s2, s0                                       // 000000001AC0: 80000002
	v_fmac_f32_e64 v0, s28, s30                                // 000000001AC4: D52B0000 00003C1C
	s_addc_u32 s1, s3, s1                                      // 000000001ACC: 82010103
	s_waitcnt lgkmcnt(0)                                       // 000000001AD0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AD4: BF870091
	v_fmac_f32_e64 v0, s29, s4                                 // 000000001AD8: D52B0000 0000081D
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001AE0: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001AE8: DC6A0000 00000001
	s_nop 0                                                    // 000000001AF0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001AF4: BFB60003
	s_endpgm                                                   // 000000001AF8: BFB00000
