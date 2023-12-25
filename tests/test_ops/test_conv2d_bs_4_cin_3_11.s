
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_3_2_9_5_3_3>:
	s_mul_hi_i32 s2, s13, 0x66666667                           // 000000001700: 9702FF0D 66666667
	s_load_b128 s[4:7], s[0:1], null                           // 000000001708: F4080100 F8000000
	s_lshr_b32 s3, s2, 31                                      // 000000001710: 85039F02
	s_ashr_i32 s2, s2, 1                                       // 000000001714: 86028102
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001718: F4040000 F8000010
	s_add_i32 s2, s2, s3                                       // 000000001720: 81020302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001724: BF870499
	s_mul_hi_i32 s3, s2, 0x38e38e39                            // 000000001728: 9703FF02 38E38E39
	s_lshr_b32 s8, s3, 31                                      // 000000001730: 85089F03
	s_ashr_i32 s3, s3, 1                                       // 000000001734: 86038103
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001738: BF8704B9
	s_add_i32 s3, s3, s8                                       // 00000000173C: 81030803
	s_mul_i32 s8, s2, 5                                        // 000000001740: 96088502
	s_mul_i32 s3, s3, 9                                        // 000000001744: 96038903
	s_sub_i32 s24, s2, s3                                      // 000000001748: 81980302
	s_mul_hi_i32 s3, s13, 0xb60b60b7                           // 00000000174C: 9703FF0D B60B60B7
	s_sub_i32 s2, s13, s8                                      // 000000001754: 8182080D
	s_mul_i32 s8, s15, 0xe7                                    // 000000001758: 9608FF0F 000000E7
	s_add_i32 s3, s3, s13                                      // 000000001760: 81030D03
	s_ashr_i32 s9, s8, 31                                      // 000000001764: 86099F08
	s_lshr_b32 s10, s3, 31                                     // 000000001768: 850A9F03
	s_ashr_i32 s3, s3, 5                                       // 00000000176C: 86038503
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001770: 84888208
	s_add_i32 s25, s3, s10                                     // 000000001774: 81190A03
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	s_add_u32 s3, s6, s8                                       // 00000000177C: 80030806
	s_mul_i32 s6, s14, 0x4d                                    // 000000001780: 9606FF0E 0000004D
	s_addc_u32 s8, s7, s9                                      // 000000001788: 82080907
	s_ashr_i32 s7, s6, 31                                      // 00000000178C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001790: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001794: 84868206
	s_add_u32 s3, s3, s6                                       // 000000001798: 80030603
	s_mul_i32 s6, s24, 7                                       // 00000000179C: 96068718
	s_addc_u32 s8, s8, s7                                      // 0000000017A0: 82080708
	s_ashr_i32 s7, s6, 31                                      // 0000000017A4: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017A8: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 0000000017AC: 84868206
	s_add_u32 s6, s3, s6                                       // 0000000017B0: 80060603
	s_addc_u32 s7, s8, s7                                      // 0000000017B4: 82070708
	s_ashr_i32 s3, s2, 31                                      // 0000000017B8: 86039F02
	s_mul_i32 s8, s14, 18                                      // 0000000017BC: 9608920E
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017C0: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017C4: BF8704B9
	s_add_u32 s6, s6, s2                                       // 0000000017C8: 80060206
	s_addc_u32 s7, s7, s3                                      // 0000000017CC: 82070307
	s_ashr_i32 s9, s8, 31                                      // 0000000017D0: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 0000000017D4: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000017D8: BF8704C9
	s_add_u32 s8, s0, s8                                       // 0000000017DC: 80080800
	s_mul_i32 s0, s25, 9                                       // 0000000017E0: 96008919
	s_addc_u32 s9, s1, s9                                      // 0000000017E4: 82090901
	s_ashr_i32 s1, s0, 31                                      // 0000000017E8: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017EC: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017F0: BF870009
	s_add_u32 s0, s8, s0                                       // 0000000017F4: 80000008
	s_addc_u32 s1, s9, s1                                      // 0000000017F8: 82010109
	s_load_b256 s[16:23], s[0:1], null                         // 0000000017FC: F40C0400 F8000000
	s_clause 0x3                                               // 000000001804: BF850003
	s_load_b64 s[8:9], s[6:7], null                            // 000000001808: F4040203 F8000000
	s_load_b32 s12, s[6:7], 0x8                                // 000000001810: F4000303 F8000008
	s_load_b64 s[10:11], s[6:7], 0x1c                          // 000000001818: F4040283 F800001C
	s_load_b32 s26, s[6:7], 0x24                               // 000000001820: F4000683 F8000024
	s_waitcnt lgkmcnt(0)                                       // 000000001828: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 00000000182C: D6130000 02002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001834: BF8700A1
	v_fmac_f32_e64 v0, s9, s17                                 // 000000001838: D52B0000 00002209
	s_load_b64 s[8:9], s[6:7], 0x38                            // 000000001840: F4040203 F8000038
	v_fmac_f32_e64 v0, s12, s18                                // 000000001848: D52B0000 0000240C
	s_mul_i32 s12, s15, 0x10e                                  // 000000001850: 960CFF0F 0000010E
	s_load_b32 s15, s[0:1], 0x20                               // 000000001858: F40003C0 F8000020
	s_ashr_i32 s13, s12, 31                                    // 000000001860: 860D9F0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001864: BF8700C1
	v_fmac_f32_e64 v0, s10, s19                                // 000000001868: D52B0000 0000260A
	s_load_b32 s10, s[6:7], 0x40                               // 000000001870: F4000283 F8000040
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001878: 8480820C
	s_mul_i32 s6, s14, 0x5a                                    // 00000000187C: 9606FF0E 0000005A
	v_fmac_f32_e64 v0, s11, s20                                // 000000001884: D52B0000 0000280B
	s_add_u32 s11, s4, s0                                      // 00000000188C: 800B0004
	s_addc_u32 s5, s5, s1                                      // 000000001890: 82050105
	s_ashr_i32 s7, s6, 31                                      // 000000001894: 86079F06
	s_mul_i32 s4, s25, 45                                      // 000000001898: 9604AD19
	v_fmac_f32_e64 v0, s26, s21                                // 00000000189C: D52B0000 00002A1A
	s_lshl_b64 s[0:1], s[6:7], 2                               // 0000000018A4: 84808206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 0000000018A8: BF8704D9
	s_add_u32 s6, s11, s0                                      // 0000000018AC: 8006000B
	s_addc_u32 s7, s5, s1                                      // 0000000018B0: 82070105
	s_waitcnt lgkmcnt(0)                                       // 0000000018B4: BF89FC07
	v_fmac_f32_e64 v0, s8, s22                                 // 0000000018B8: D52B0000 00002C08
	s_ashr_i32 s5, s4, 31                                      // 0000000018C0: 86059F04
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000018C4: 84808204
	s_mul_i32 s4, s24, 5                                       // 0000000018C8: 96048518
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000018CC: BF8700C1
	v_fmac_f32_e64 v0, s9, s23                                 // 0000000018D0: D52B0000 00002E09
	s_add_u32 s6, s6, s0                                       // 0000000018D8: 80060006
	s_addc_u32 s7, s7, s1                                      // 0000000018DC: 82070107
	s_ashr_i32 s5, s4, 31                                      // 0000000018E0: 86059F04
	v_fmac_f32_e64 v0, s10, s15                                // 0000000018E4: D52B0000 00001E0A
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000018EC: 84808204
	v_mov_b32_e32 v1, 0                                        // 0000000018F0: 7E020280
	s_add_u32 s0, s6, s0                                       // 0000000018F4: 80000006
	s_addc_u32 s1, s7, s1                                      // 0000000018F8: 82010107
	v_max_f32_e32 v0, 0, v0                                    // 0000000018FC: 20000080
	s_add_u32 s0, s0, s2                                       // 000000001900: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001904: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001908: DC6A0000 00000001
	s_nop 0                                                    // 000000001910: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001914: BFB60003
	s_endpgm                                                   // 000000001918: BFB00000
