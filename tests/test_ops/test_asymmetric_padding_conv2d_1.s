
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_5_5_2_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s15, 3                                       // 000000001714: 9608830F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF89FC07
	s_add_u32 s10, s6, 0xffffffe0                              // 000000001720: 800AFF06 FFFFFFE0
	s_addc_u32 s3, s7, -1                                      // 000000001728: 8203C107
	s_add_i32 s6, s15, -1                                      // 00000000172C: 8106C10F
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_cmp_gt_i32 s6, 0                                         // 000000001734: BF028006
	s_cselect_b32 s13, -1, 0                                   // 000000001738: 980D80C1
	s_cmp_lt_i32 s15, 5                                        // 00000000173C: BF04850F
	s_cselect_b32 s16, -1, 0                                   // 000000001740: 981080C1
	s_add_i32 s7, s14, -1                                      // 000000001744: 8107C10E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 000000001748: BF8704C9
	s_cmp_gt_i32 s7, 0                                         // 00000000174C: BF028007
	s_cselect_b32 s6, -1, 0                                    // 000000001750: 980680C1
	s_cmp_lt_i32 s14, 5                                        // 000000001754: BF04850E
	s_cselect_b32 s11, -1, 0                                   // 000000001758: 980B80C1
	s_and_b32 s6, s11, s6                                      // 00000000175C: 8B06060B
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_and_b32 s11, s13, s6                                     // 000000001764: 8B0B060D
	s_and_b32 s12, s16, s11                                    // 000000001768: 8B0C0B10
	s_mov_b32 s11, 0                                           // 00000000176C: BE8B0080
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 000000001770: 916A0C7E
	s_mov_b32 s12, 0                                           // 000000001774: BE8C0080
	s_cbranch_vccnz 10                                         // 000000001778: BFA4000A <r_5_5_2_2+0xa4>
	s_lshl_b64 s[18:19], s[8:9], 2                             // 00000000177C: 84928208
	s_ashr_i32 s15, s14, 31                                    // 000000001780: 860F9F0E
	s_add_u32 s12, s10, s18                                    // 000000001784: 800C120A
	s_addc_u32 s17, s3, s19                                    // 000000001788: 82111303
	s_lshl_b64 s[18:19], s[14:15], 2                           // 00000000178C: 8492820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001790: BF870009
	s_add_u32 s18, s12, s18                                    // 000000001794: 8012120C
	s_addc_u32 s19, s17, s19                                   // 000000001798: 82131311
	s_load_b32 s12, s[18:19], null                             // 00000000179C: F4000309 F8000000
	s_cmp_lt_u32 s7, 3                                         // 0000000017A4: BF0A8307
	s_cselect_b32 s7, -1, 0                                    // 0000000017A8: 980780C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017AC: BF870499
	s_and_b32 s13, s13, s7                                     // 0000000017B0: 8B0D070D
	s_and_b32 s13, s16, s13                                    // 0000000017B4: 8B0D0D10
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B8: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 0000000017BC: 916A0D7E
	s_cbranch_vccnz 11                                         // 0000000017C0: BFA4000B <r_5_5_2_2+0xf0>
	s_lshl_b64 s[16:17], s[8:9], 2                             // 0000000017C4: 84908208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017C8: BF8704B9
	s_add_u32 s11, s10, s16                                    // 0000000017CC: 800B100A
	s_addc_u32 s13, s3, s17                                    // 0000000017D0: 820D1103
	s_ashr_i32 s15, s14, 31                                    // 0000000017D4: 860F9F0E
	s_lshl_b64 s[16:17], s[14:15], 2                           // 0000000017D8: 8490820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017DC: BF870009
	s_add_u32 s16, s11, s16                                    // 0000000017E0: 8010100B
	s_addc_u32 s17, s13, s17                                   // 0000000017E4: 8211110D
	s_load_b32 s11, s[16:17], 0x4                              // 0000000017E8: F40002C8 F8000004
	s_add_i32 s13, s2, 1                                       // 0000000017F0: 810D8102
	s_cmp_gt_i32 s2, 0                                         // 0000000017F4: BF028002
	s_cselect_b32 s16, -1, 0                                   // 0000000017F8: 981080C1
	s_cmp_lt_i32 s13, 5                                        // 0000000017FC: BF04850D
	s_mov_b32 s13, 0                                           // 000000001800: BE8D0080
	s_cselect_b32 s17, -1, 0                                   // 000000001804: 981180C1
	s_and_b32 s6, s16, s6                                      // 000000001808: 8B060610
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000180C: BF870499
	s_and_b32 s6, s17, s6                                      // 000000001810: 8B060611
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000001814: 916A067E
	s_cbranch_vccnz 10                                         // 000000001818: BFA4000A <r_5_5_2_2+0x144>
	s_lshl_b64 s[18:19], s[8:9], 2                             // 00000000181C: 84928208
	s_ashr_i32 s15, s14, 31                                    // 000000001820: 860F9F0E
	s_add_u32 s6, s10, s18                                     // 000000001824: 8006120A
	s_addc_u32 s13, s3, s19                                    // 000000001828: 820D1303
	s_lshl_b64 s[18:19], s[14:15], 2                           // 00000000182C: 8492820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001830: BF870009
	s_add_u32 s18, s6, s18                                     // 000000001834: 80121206
	s_addc_u32 s19, s13, s19                                   // 000000001838: 8213130D
	s_load_b32 s13, s[18:19], 0xc                              // 00000000183C: F4000349 F800000C
	s_and_b32 s6, s16, s7                                      // 000000001844: 8B060710
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001848: BF870499
	s_and_b32 s6, s17, s6                                      // 00000000184C: 8B060611
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001850: 8B6A067E
	s_cbranch_vccnz 9                                          // 000000001854: BFA40009 <r_5_5_2_2+0x17c>
	s_ashr_i32 s15, s14, 31                                    // 000000001858: 860F9F0E
	s_mov_b32 s16, 0                                           // 00000000185C: BE900080
	s_clause 0x1                                               // 000000001860: BF850001
	s_load_b64 s[6:7], s[0:1], null                            // 000000001864: F4040180 F8000000
	s_load_b32 s17, s[0:1], 0x8                                // 00000000186C: F4000440 F8000008
	s_cbranch_execz 6                                          // 000000001874: BFA50006 <r_5_5_2_2+0x190>
	s_branch 16                                                // 000000001878: BFA00010 <r_5_5_2_2+0x1bc>
	s_clause 0x1                                               // 00000000187C: BF850001
	s_load_b64 s[6:7], s[0:1], null                            // 000000001880: F4040180 F8000000
	s_load_b32 s17, s[0:1], 0x8                                // 000000001888: F4000440 F8000008
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001890: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001894: BF8704B9
	s_add_u32 s10, s10, s8                                     // 000000001898: 800A080A
	s_addc_u32 s3, s3, s9                                      // 00000000189C: 82030903
	s_ashr_i32 s15, s14, 31                                    // 0000000018A0: 860F9F0E
	s_lshl_b64 s[8:9], s[14:15], 2                             // 0000000018A4: 8488820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000018A8: BF870009
	s_add_u32 s8, s10, s8                                      // 0000000018AC: 8008080A
	s_addc_u32 s9, s3, s9                                      // 0000000018B0: 82090903
	s_load_b32 s16, s[8:9], 0x10                               // 0000000018B4: F4000404 F8000010
	s_load_b32 s3, s[0:1], 0xc                                 // 0000000018BC: F40000C0 F800000C
	s_waitcnt lgkmcnt(0)                                       // 0000000018C4: BF89FC07
	v_fma_f32 v0, s12, s6, 0                                   // 0000000018C8: D6130000 02000C0C
	s_mul_i32 s0, s2, 5                                        // 0000000018D0: 96008502
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018D4: BF870099
	s_ashr_i32 s1, s0, 31                                      // 0000000018D8: 86019F00
	v_fmac_f32_e64 v0, s11, s7                                 // 0000000018DC: D52B0000 00000E0B
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000018E4: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018E8: BF870099
	s_add_u32 s2, s4, s0                                       // 0000000018EC: 80020004
	v_fmac_f32_e64 v0, s13, s17                                // 0000000018F0: D52B0000 0000220D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000018F8: BF870141
	v_fmac_f32_e64 v0, s16, s3                                 // 0000000018FC: D52B0000 00000610
	v_mov_b32_e32 v1, 0                                        // 000000001904: 7E020280
	s_addc_u32 s3, s5, s1                                      // 000000001908: 82030105
	s_lshl_b64 s[0:1], s[14:15], 2                             // 00000000190C: 8480820E
	v_max_f32_e32 v0, 0, v0                                    // 000000001910: 20000080
	s_add_u32 s0, s2, s0                                       // 000000001914: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001918: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 00000000191C: DC6A0000 00000001
	s_nop 0                                                    // 000000001924: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001928: BFB60003
	s_endpgm                                                   // 00000000192C: BFB00000
