
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_5_4_2_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_add_i32 s13, s14, -1                                     // 00000000171C: 810DC10E
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001720: 8480820E
	s_mul_i32 s10, s2, 3                                       // 000000001724: 960A8302
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s3, s6, s0                                       // 00000000172C: 80030006
	s_addc_u32 s6, s7, s1                                      // 000000001730: 82060107
	s_add_u32 s12, s3, 0xffffffe0                              // 000000001734: 800CFF03 FFFFFFE0
	s_addc_u32 s3, s6, -1                                      // 00000000173C: 8203C106
	s_add_i32 s6, s2, -1                                       // 000000001740: 8106C102
	s_ashr_i32 s11, s10, 31                                    // 000000001744: 860B9F0A
	s_cmp_gt_i32 s6, 0                                         // 000000001748: BF028006
	s_cselect_b32 s7, -1, 0                                    // 00000000174C: 980780C1
	s_cmp_lt_i32 s2, 5                                         // 000000001750: BF048502
	s_cselect_b32 s16, -1, 0                                   // 000000001754: 981080C1
	s_cmp_gt_i32 s13, 0                                        // 000000001758: BF02800D
	s_cselect_b32 s6, -1, 0                                    // 00000000175C: 980680C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_and_b32 s13, s7, s6                                      // 000000001764: 8B0D0607
	s_and_b32 s15, s16, s13                                    // 000000001768: 8B0F0D10
	s_mov_b32 s13, 0                                           // 00000000176C: BE8D0080
	s_and_not1_b32 vcc_lo, exec_lo, s15                        // 000000001770: 916A0F7E
	s_mov_b32 s15, 0                                           // 000000001774: BE8F0080
	s_cbranch_vccnz 6                                          // 000000001778: BFA40006 <r_5_4_2_2+0x94>
	s_lshl_b64 s[18:19], s[10:11], 2                           // 00000000177C: 8492820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s18, s12, s18                                    // 000000001784: 8012120C
	s_addc_u32 s19, s3, s19                                    // 000000001788: 82131303
	s_load_b32 s15, s[18:19], null                             // 00000000178C: F40003C9 F8000000
	s_cmp_gt_i32 s14, 0                                        // 000000001794: BF02800E
	s_cselect_b32 s14, -1, 0                                   // 000000001798: 980E80C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000179C: BF870499
	s_and_b32 s7, s7, s14                                      // 0000000017A0: 8B070E07
	s_and_b32 s7, s16, s7                                      // 0000000017A4: 8B070710
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A8: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 0000000017AC: 916A077E
	s_cbranch_vccnz 6                                          // 0000000017B0: BFA40006 <r_5_4_2_2+0xcc>
	s_lshl_b64 s[16:17], s[10:11], 2                           // 0000000017B4: 8490820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B8: BF870009
	s_add_u32 s16, s12, s16                                    // 0000000017BC: 8010100C
	s_addc_u32 s17, s3, s17                                    // 0000000017C0: 82111103
	s_load_b32 s13, s[16:17], 0x4                              // 0000000017C4: F4000348 F8000004
	s_add_i32 s7, s2, 1                                        // 0000000017CC: 81078102
	s_cmp_gt_i32 s2, 0                                         // 0000000017D0: BF028002
	s_mov_b32 s16, 0                                           // 0000000017D4: BE900080
	s_cselect_b32 s19, -1, 0                                   // 0000000017D8: 981380C1
	s_cmp_lt_i32 s7, 5                                         // 0000000017DC: BF048507
	s_mov_b32 s17, 0                                           // 0000000017E0: BE910080
	s_cselect_b32 s20, -1, 0                                   // 0000000017E4: 981480C1
	s_and_b32 s6, s19, s6                                      // 0000000017E8: 8B060613
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017EC: BF870499
	s_and_b32 s6, s20, s6                                      // 0000000017F0: 8B060614
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 0000000017F4: 916A067E
	s_cbranch_vccnz 6                                          // 0000000017F8: BFA40006 <r_5_4_2_2+0x114>
	s_lshl_b64 s[6:7], s[10:11], 2                             // 0000000017FC: 8486820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001800: BF870009
	s_add_u32 s6, s12, s6                                      // 000000001804: 8006060C
	s_addc_u32 s7, s3, s7                                      // 000000001808: 82070703
	s_load_b32 s17, s[6:7], 0xc                                // 00000000180C: F4000443 F800000C
	s_clause 0x1                                               // 000000001814: BF850001
	s_load_b64 s[6:7], s[8:9], null                            // 000000001818: F4040184 F8000000
	s_load_b32 s18, s[8:9], 0x8                                // 000000001820: F4000484 F8000008
	s_and_b32 s14, s19, s14                                    // 000000001828: 8B0E0E13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000182C: BF870499
	s_and_b32 s14, s20, s14                                    // 000000001830: 8B0E0E14
	s_and_not1_b32 vcc_lo, exec_lo, s14                        // 000000001834: 916A0E7E
	s_cbranch_vccnz 6                                          // 000000001838: BFA40006 <r_5_4_2_2+0x154>
	s_lshl_b64 s[10:11], s[10:11], 2                           // 00000000183C: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001840: BF870009
	s_add_u32 s10, s12, s10                                    // 000000001844: 800A0A0C
	s_addc_u32 s11, s3, s11                                    // 000000001848: 820B0B03
	s_load_b32 s16, s[10:11], 0x10                             // 00000000184C: F4000405 F8000010
	s_load_b32 s3, s[8:9], 0xc                                 // 000000001854: F40000C4 F800000C
	s_waitcnt lgkmcnt(0)                                       // 00000000185C: BF89FC07
	v_fma_f32 v0, s15, s6, 0                                   // 000000001860: D6130000 02000C0F
	s_lshl_b32 s2, s2, 2                                       // 000000001868: 84028202
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000186C: BF870091
	v_fmac_f32_e64 v0, s13, s7                                 // 000000001870: D52B0000 00000E0D
	v_fmac_f32_e64 v0, s17, s18                                // 000000001878: D52B0000 00002411
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001880: BF870141
	v_fmac_f32_e64 v0, s16, s3                                 // 000000001884: D52B0000 00000610
	s_ashr_i32 s3, s2, 31                                      // 00000000188C: 86039F02
	v_mov_b32_e32 v1, 0                                        // 000000001890: 7E020280
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001894: 84828202
	v_max_f32_e32 v0, 0, v0                                    // 000000001898: 20000080
	s_add_u32 s2, s4, s2                                       // 00000000189C: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000018A0: 82030305
	s_add_u32 s0, s2, s0                                       // 0000000018A4: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000018A8: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000018AC: DC6A0000 00000001
	s_nop 0                                                    // 0000000018B4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018B8: BFB60003
	s_endpgm                                                   // 0000000018BC: BFB00000
