
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_9_6_3_3_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x2aaaaaab                           // 000000001714: 9708FF0D 2AAAAAAB
	s_mul_i32 s2, s15, 0xe7                                    // 00000000171C: 9602FF0F 000000E7
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s9, s8, s9                                       // 00000000172C: 81090908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s8, s9, 6                                        // 000000001734: 96088609
	s_mul_i32 s10, s9, 7                                       // 000000001738: 960A8709
	s_sub_i32 s12, s13, s8                                     // 00000000173C: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_add_u32 s9, s6, s2                                       // 000000001758: 80090206
	s_addc_u32 s7, s7, s3                                      // 00000000175C: 82070307
	s_ashr_i32 s13, s12, 31                                    // 000000001760: 860D9F0C
	s_mul_i32 s6, s14, 18                                      // 000000001764: 9606920E
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001768: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s10, s9, s2                                      // 000000001770: 800A0209
	s_addc_u32 s11, s7, s3                                     // 000000001774: 820B0307
	s_ashr_i32 s7, s6, 31                                      // 000000001778: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000177C: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001784: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001788: 82010701
	s_load_b512 s[16:31], s[0:1], null                         // 00000000178C: F4100400 F8000000
	s_clause 0x3                                               // 000000001794: BF850003
	s_load_b64 s[6:7], s[10:11], null                          // 000000001798: F4040185 F8000000
	s_load_b64 s[12:13], s[10:11], 0x1c                        // 0000000017A0: F4040305 F800001C
	s_load_b64 s[34:35], s[10:11], 0x38                        // 0000000017A8: F4040885 F8000038
	s_load_b64 s[36:37], s[10:11], 0x134                       // 0000000017B0: F4040905 F8000134
	s_waitcnt lgkmcnt(0)                                       // 0000000017B8: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 0000000017BC: D6130000 02002006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017C4: BF8700A1
	v_fmac_f32_e64 v0, s7, s17                                 // 0000000017C8: D52B0000 00002207
	s_load_b64 s[6:7], s[10:11], 0x150                         // 0000000017D0: F4040185 F8000150
	v_fmac_f32_e64 v0, s12, s18                                // 0000000017D8: D52B0000 0000240C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017E0: BF870001
	v_fmac_f32_e64 v0, s13, s19                                // 0000000017E4: D52B0000 0000260D
	s_clause 0x2                                               // 0000000017EC: BF850002
	s_load_b64 s[12:13], s[10:11], 0x16c                       // 0000000017F0: F4040305 F800016C
	s_load_b64 s[16:17], s[10:11], 0x268                       // 0000000017F8: F4040405 F8000268
	s_load_b64 s[18:19], s[10:11], 0x284                       // 000000001800: F4040485 F8000284
	v_fmac_f32_e64 v0, s34, s20                                // 000000001808: D52B0000 00002822
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001810: BF870091
	v_fmac_f32_e64 v0, s35, s21                                // 000000001814: D52B0000 00002A23
	v_fmac_f32_e64 v0, s36, s22                                // 00000000181C: D52B0000 00002C24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001824: BF8700A1
	v_fmac_f32_e64 v0, s37, s23                                // 000000001828: D52B0000 00002E25
	s_waitcnt lgkmcnt(0)                                       // 000000001830: BF89FC07
	v_fmac_f32_e64 v0, s6, s24                                 // 000000001834: D52B0000 00003006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 00000000183C: BF8704C1
	v_fmac_f32_e64 v0, s7, s25                                 // 000000001840: D52B0000 00003207
	s_load_b64 s[6:7], s[10:11], 0x2a0                         // 000000001848: F4040185 F80002A0
	s_load_b64 s[0:1], s[0:1], 0x40                            // 000000001850: F4040000 F8000040
	s_mul_i32 s10, s15, 0x144                                  // 000000001858: 960AFF0F 00000144
	s_ashr_i32 s11, s10, 31                                    // 000000001860: 860B9F0A
	v_fmac_f32_e64 v0, s12, s26                                // 000000001864: D52B0000 0000340C
	s_lshl_b64 s[10:11], s[10:11], 2                           // 00000000186C: 848A820A
	s_mul_i32 s12, s14, 54                                     // 000000001870: 960CB60E
	s_add_u32 s9, s4, s10                                      // 000000001874: 80090A04
	s_addc_u32 s10, s5, s11                                    // 000000001878: 820A0B05
	v_fmac_f32_e64 v0, s13, s27                                // 00000000187C: D52B0000 0000360D
	s_ashr_i32 s13, s12, 31                                    // 000000001884: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870099
	s_lshl_b64 s[4:5], s[12:13], 2                             // 00000000188C: 8484820C
	v_fmac_f32_e64 v0, s16, s28                                // 000000001890: D52B0000 00003810
	s_add_u32 s4, s9, s4                                       // 000000001898: 80040409
	s_addc_u32 s5, s10, s5                                     // 00000000189C: 8205050A
	s_ashr_i32 s9, s8, 31                                      // 0000000018A0: 86099F08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A4: BF870091
	v_fmac_f32_e64 v0, s17, s29                                // 0000000018A8: D52B0000 00003A11
	v_fmac_f32_e64 v0, s18, s30                                // 0000000018B0: D52B0000 00003C12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018B8: BF8700A1
	v_fmac_f32_e64 v0, s19, s31                                // 0000000018BC: D52B0000 00003E13
	s_waitcnt lgkmcnt(0)                                       // 0000000018C4: BF89FC07
	v_fmac_f32_e64 v0, s6, s0                                  // 0000000018C8: D52B0000 00000006
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018D0: BF870001
	v_fmac_f32_e64 v0, s7, s1                                  // 0000000018D4: D52B0000 00000207
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000018DC: 84808208
	v_mov_b32_e32 v1, 0                                        // 0000000018E0: 7E020280
	s_add_u32 s0, s4, s0                                       // 0000000018E4: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000018E8: 82010105
	v_max_f32_e32 v0, 0, v0                                    // 0000000018EC: 20000080
	s_add_u32 s0, s0, s2                                       // 0000000018F0: 80000200
	s_addc_u32 s1, s1, s3                                      // 0000000018F4: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000018F8: DC6A0000 00000001
	s_nop 0                                                    // 000000001900: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001904: BFB60003
	s_endpgm                                                   // 000000001908: BFB00000
