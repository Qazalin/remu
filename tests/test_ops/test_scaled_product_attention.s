
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_256_16_16_64>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s8, s15, 10                                     // 000000001714: 84088A0F
	v_mov_b32_e32 v0, 0                                        // 000000001718: 7E000280
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_mov_b32 s2, s13                                          // 000000001720: BE82000D
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s3, s6, s8                                       // 00000000172C: 80030806
	s_addc_u32 s10, s7, s9                                     // 000000001730: 820A0907
	s_lshl_b32 s6, s14, 6                                      // 000000001734: 8406860E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_ashr_i32 s7, s6, 31                                      // 00000000173C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001740: 84868206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001744: BF8704D9
	s_add_u32 s3, s3, s6                                       // 000000001748: 80030603
	s_addc_u32 s6, s10, s7                                     // 00000000174C: 8206070A
	s_add_u32 s7, s0, s8                                       // 000000001750: 80070800
	s_addc_u32 s8, s1, s9                                      // 000000001754: 82080901
	s_lshl_b32 s0, s13, 6                                      // 000000001758: 8400860D
	s_ashr_i32 s1, s0, 31                                      // 00000000175C: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001760: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001764: 84808200
	s_add_u32 s7, s7, s0                                       // 000000001768: 80070007
	s_addc_u32 s8, s8, s1                                      // 00000000176C: 82080108
	s_mov_b64 s[0:1], 0                                        // 000000001770: BE800180
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001774: BF870009
	s_add_u32 s10, s3, s0                                      // 000000001778: 800A0003
	s_addc_u32 s11, s6, s1                                     // 00000000177C: 820B0106
	s_add_u32 s12, s7, s0                                      // 000000001780: 800C0007
	s_addc_u32 s13, s8, s1                                     // 000000001784: 820D0108
	s_load_b512 s[16:31], s[10:11], null                       // 000000001788: F4100405 F8000000
	s_load_b512 s[36:51], s[12:13], null                       // 000000001790: F4100906 F8000000
	s_add_u32 s0, s0, 64                                       // 000000001798: 8000C000
	s_addc_u32 s1, s1, 0                                       // 00000000179C: 82018001
	s_cmpk_eq_i32 s0, 0x100                                    // 0000000017A0: B1800100
	s_waitcnt lgkmcnt(0)                                       // 0000000017A4: BF89FC07
	v_fmac_f32_e64 v0, s16, s36                                // 0000000017A8: D52B0000 00004810
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B0: BF870091
	v_fmac_f32_e64 v0, s17, s37                                // 0000000017B4: D52B0000 00004A11
	v_fmac_f32_e64 v0, s18, s38                                // 0000000017BC: D52B0000 00004C12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_fmac_f32_e64 v0, s19, s39                                // 0000000017C8: D52B0000 00004E13
	v_fmac_f32_e64 v0, s20, s40                                // 0000000017D0: D52B0000 00005014
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_fmac_f32_e64 v0, s21, s41                                // 0000000017DC: D52B0000 00005215
	v_fmac_f32_e64 v0, s22, s42                                // 0000000017E4: D52B0000 00005416
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017EC: BF870091
	v_fmac_f32_e64 v0, s23, s43                                // 0000000017F0: D52B0000 00005617
	v_fmac_f32_e64 v0, s24, s44                                // 0000000017F8: D52B0000 00005818
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001800: BF870091
	v_fmac_f32_e64 v0, s25, s45                                // 000000001804: D52B0000 00005A19
	v_fmac_f32_e64 v0, s26, s46                                // 00000000180C: D52B0000 00005C1A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001814: BF870091
	v_fmac_f32_e64 v0, s27, s47                                // 000000001818: D52B0000 00005E1B
	v_fmac_f32_e64 v0, s28, s48                                // 000000001820: D52B0000 0000601C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001828: BF870091
	v_fmac_f32_e64 v0, s29, s49                                // 00000000182C: D52B0000 0000621D
	v_fmac_f32_e64 v0, s30, s50                                // 000000001834: D52B0000 0000641E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000183C: BF870001
	v_fmac_f32_e64 v0, s31, s51                                // 000000001840: D52B0000 0000661F
	s_cbranch_scc0 65482                                       // 000000001848: BFA1FFCA <r_256_16_16_64+0x74>
	s_lshl_b32 s0, s15, 8                                      // 00000000184C: 8400880F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001850: BF8704A1
	v_dual_mul_f32 v0, 0x3e000000, v0 :: v_dual_mov_b32 v1, 0  // 000000001854: C8D000FF 00000080 3E000000
	s_ashr_i32 s1, s0, 31                                      // 000000001860: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001864: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001868: BF8704B9
	s_add_u32 s3, s4, s0                                       // 00000000186C: 80030004
	s_addc_u32 s4, s5, s1                                      // 000000001870: 82040105
	s_lshl_b32 s0, s14, 4                                      // 000000001874: 8400840E
	s_ashr_i32 s1, s0, 31                                      // 000000001878: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000187C: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001880: 84808200
	s_add_u32 s5, s3, s0                                       // 000000001884: 80050003
	s_addc_u32 s4, s4, s1                                      // 000000001888: 82040104
	s_ashr_i32 s3, s2, 31                                      // 00000000188C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001890: BF870499
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001894: 84808202
	s_add_u32 s0, s5, s0                                       // 000000001898: 80000005
	s_addc_u32 s1, s4, s1                                      // 00000000189C: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 0000000018A0: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018AC: BFB60003
	s_endpgm                                                   // 0000000018B0: BFB00000
