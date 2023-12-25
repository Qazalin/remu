
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_10_5_3_2_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s3, s13, 0x66666667                           // 000000001714: 9703FF0D 66666667
	s_mul_i32 s2, s15, 0xe7                                    // 00000000171C: 9602FF0F 000000E7
	s_lshr_b32 s8, s3, 31                                      // 000000001724: 85089F03
	s_ashr_i32 s9, s3, 1                                       // 000000001728: 86098103
	s_ashr_i32 s3, s2, 31                                      // 00000000172C: 86039F02
	s_add_i32 s9, s9, s8                                       // 000000001730: 81090809
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001734: 84828202
	s_mul_i32 s8, s9, 5                                        // 000000001738: 96088509
	s_mul_i32 s10, s9, 7                                       // 00000000173C: 960A8709
	s_sub_i32 s12, s13, s8                                     // 000000001740: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001748: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000174C: 82070307
	s_ashr_i32 s11, s10, 31                                    // 000000001750: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001758: 8482820A
	s_add_u32 s9, s6, s2                                       // 00000000175C: 80090206
	s_addc_u32 s7, s7, s3                                      // 000000001760: 82070307
	s_ashr_i32 s13, s12, 31                                    // 000000001764: 860D9F0C
	s_mul_i32 s6, s14, 18                                      // 000000001768: 9606920E
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000176C: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001770: BF8704B9
	s_add_u32 s10, s9, s2                                      // 000000001774: 800A0209
	s_addc_u32 s11, s7, s3                                     // 000000001778: 820B0307
	s_ashr_i32 s7, s6, 31                                      // 00000000177C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001780: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001788: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000178C: 82010701
	s_load_b512 s[16:31], s[0:1], null                         // 000000001790: F4100400 F8000000
	s_clause 0x3                                               // 000000001798: BF850003
	s_load_b64 s[6:7], s[10:11], null                          // 00000000179C: F4040185 F8000000
	s_load_b32 s9, s[10:11], 0x8                               // 0000000017A4: F4000245 F8000008
	s_load_b64 s[12:13], s[10:11], 0x1c                        // 0000000017AC: F4040305 F800001C
	s_load_b32 s33, s[10:11], 0x24                             // 0000000017B4: F4000845 F8000024
	s_waitcnt lgkmcnt(0)                                       // 0000000017BC: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 0000000017C0: D6130000 02002006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017C8: BF8700C1
	v_fmac_f32_e64 v0, s7, s17                                 // 0000000017CC: D52B0000 00002207
	s_clause 0x1                                               // 0000000017D4: BF850001
	s_load_b64 s[6:7], s[10:11], 0x150                         // 0000000017D8: F4040185 F8000150
	s_load_b64 s[16:17], s[10:11], 0x134                       // 0000000017E0: F4040405 F8000134
	v_fmac_f32_e64 v0, s9, s18                                 // 0000000017E8: D52B0000 00002409
	s_clause 0x1                                               // 0000000017F0: BF850001
	s_load_b32 s9, s[10:11], 0x13c                             // 0000000017F4: F4000245 F800013C
	s_load_b32 s18, s[10:11], 0x158                            // 0000000017FC: F4000485 F8000158
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001804: BF870091
	v_fmac_f32_e64 v0, s12, s19                                // 000000001808: D52B0000 0000260C
	v_fmac_f32_e64 v0, s13, s20                                // 000000001810: D52B0000 0000280D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001818: BF8700A1
	v_fmac_f32_e64 v0, s33, s21                                // 00000000181C: D52B0000 00002A21
	s_waitcnt lgkmcnt(0)                                       // 000000001824: BF89FC07
	v_fmac_f32_e64 v0, s16, s22                                // 000000001828: D52B0000 00002C10
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001830: BF870001
	v_fmac_f32_e64 v0, s17, s23                                // 000000001834: D52B0000 00002E11
	s_clause 0x1                                               // 00000000183C: BF850001
	s_load_b64 s[12:13], s[10:11], 0x284                       // 000000001840: F4040305 F8000284
	s_load_b64 s[16:17], s[10:11], 0x268                       // 000000001848: F4040405 F8000268
	s_load_b64 s[0:1], s[0:1], 0x40                            // 000000001850: F4040000 F8000040
	v_fmac_f32_e64 v0, s9, s24                                 // 000000001858: D52B0000 00003009
	s_load_b32 s9, s[10:11], 0x270                             // 000000001860: F4000245 F8000270
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001868: BF8700A1
	v_fmac_f32_e64 v0, s6, s25                                 // 00000000186C: D52B0000 00003206
	s_mul_i32 s6, s15, 0x12c                                   // 000000001874: 9606FF0F 0000012C
	v_fmac_f32_e64 v0, s7, s26                                 // 00000000187C: D52B0000 00003407
	s_ashr_i32 s7, s6, 31                                      // 000000001884: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870099
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000188C: 84868206
	v_fmac_f32_e64 v0, s18, s27                                // 000000001890: D52B0000 00003612
	s_add_u32 s6, s4, s6                                       // 000000001898: 80060604
	s_addc_u32 s7, s5, s7                                      // 00000000189C: 82070705
	s_waitcnt lgkmcnt(0)                                       // 0000000018A0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000018A4: BF8704B1
	v_fmac_f32_e64 v0, s16, s28                                // 0000000018A8: D52B0000 00003810
	s_load_b32 s16, s[10:11], 0x28c                            // 0000000018B0: F4000405 F800028C
	s_mul_i32 s10, s14, 50                                     // 0000000018B8: 960AB20E
	s_ashr_i32 s11, s10, 31                                    // 0000000018BC: 860B9F0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000018C0: BF8704A1
	v_fmac_f32_e64 v0, s17, s29                                // 0000000018C4: D52B0000 00003A11
	s_lshl_b64 s[4:5], s[10:11], 2                             // 0000000018CC: 8484820A
	s_add_u32 s4, s6, s4                                       // 0000000018D0: 80040406
	s_addc_u32 s5, s7, s5                                      // 0000000018D4: 82050507
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018D8: BF8700A1
	v_fmac_f32_e64 v0, s9, s30                                 // 0000000018DC: D52B0000 00003C09
	s_ashr_i32 s9, s8, 31                                      // 0000000018E4: 86099F08
	v_fmac_f32_e64 v0, s12, s31                                // 0000000018E8: D52B0000 00003E0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018F0: BF8700A1
	v_fmac_f32_e64 v0, s13, s0                                 // 0000000018F4: D52B0000 0000000D
	s_waitcnt lgkmcnt(0)                                       // 0000000018FC: BF89FC07
	v_fmac_f32_e64 v0, s16, s1                                 // 000000001900: D52B0000 00000210
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001908: 84808208
	v_mov_b32_e32 v1, 0                                        // 00000000190C: 7E020280
	s_add_u32 s0, s4, s0                                       // 000000001910: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001914: 82010105
	v_max_f32_e32 v0, 0, v0                                    // 000000001918: 20000080
	s_add_u32 s0, s0, s2                                       // 00000000191C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001920: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001924: DC6A0000 00000001
	s_nop 0                                                    // 00000000192C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001930: BFB60003
	s_endpgm                                                   // 000000001934: BFB00000
