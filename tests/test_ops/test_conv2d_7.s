
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_10_3_3_2_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001730: 82090907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001738: 84868202
	s_mul_i32 s2, s15, 30                                      // 00000000173C: 96029E0F
	s_add_u32 s12, s8, s6                                      // 000000001740: 800C0608
	s_addc_u32 s13, s9, s7                                     // 000000001744: 820D0709
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[34:35], s[2:3], 2                             // 000000001750: 84A28202
	s_add_u32 s36, s0, s34                                     // 000000001754: 80242200
	s_addc_u32 s37, s1, s35                                    // 000000001758: 82252301
	s_load_b512 s[16:31], s[36:37], null                       // 00000000175C: F4100412 F8000000
	s_clause 0x2                                               // 000000001764: BF850002
	s_load_b128 s[0:3], s[12:13], null                         // 000000001768: F4080006 F8000000
	s_load_b32 s15, s[12:13], 0x10                             // 000000001770: F40003C6 F8000010
	s_load_b128 s[8:11], s[12:13], 0x1c                        // 000000001778: F4080206 F800001C
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_fma_f32 v0, s0, s16, 0                                   // 000000001784: D6130000 02002000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000178C: BF870091
	v_fmac_f32_e64 v0, s1, s17                                 // 000000001790: D52B0000 00002201
	v_fmac_f32_e64 v0, s2, s18                                 // 000000001798: D52B0000 00002402
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017A0: BF8700A1
	v_fmac_f32_e64 v0, s3, s19                                 // 0000000017A4: D52B0000 00002603
	s_load_b128 s[0:3], s[12:13], 0x134                        // 0000000017AC: F4080006 F8000134
	v_fmac_f32_e64 v0, s15, s20                                // 0000000017B4: D52B0000 0000280F
	s_load_b32 s15, s[12:13], 0x2c                             // 0000000017BC: F40003C6 F800002C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_fmac_f32_e64 v0, s8, s21                                 // 0000000017C8: D52B0000 00002A08
	v_fmac_f32_e64 v0, s9, s22                                 // 0000000017D0: D52B0000 00002C09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017D8: BF8700A1
	v_fmac_f32_e64 v0, s10, s23                                // 0000000017DC: D52B0000 00002E0A
	s_load_b256 s[16:23], s[36:37], 0x40                       // 0000000017E4: F40C0412 F8000040
	v_fmac_f32_e64 v0, s11, s24                                // 0000000017EC: D52B0000 0000300B
	s_load_b128 s[8:11], s[12:13], 0x150                       // 0000000017F4: F4080206 F8000150
	s_waitcnt lgkmcnt(0)                                       // 0000000017FC: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001800: BF8700A1
	v_fmac_f32_e64 v0, s15, s25                                // 000000001804: D52B0000 0000320F
	s_load_b32 s15, s[12:13], 0x144                            // 00000000180C: F40003C6 F8000144
	v_fmac_f32_e64 v0, s0, s26                                 // 000000001814: D52B0000 00003400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000181C: BF870091
	v_fmac_f32_e64 v0, s1, s27                                 // 000000001820: D52B0000 00003601
	v_fmac_f32_e64 v0, s2, s28                                 // 000000001828: D52B0000 00003802
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001830: BF8700B1
	v_fmac_f32_e64 v0, s3, s29                                 // 000000001834: D52B0000 00003A03
	s_load_b128 s[0:3], s[12:13], 0x268                        // 00000000183C: F4080006 F8000268
	s_waitcnt lgkmcnt(0)                                       // 000000001844: BF89FC07
	v_fmac_f32_e64 v0, s15, s30                                // 000000001848: D52B0000 00003C0F
	s_load_b32 s15, s[12:13], 0x160                            // 000000001850: F40003C6 F8000160
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001858: BF870091
	v_fmac_f32_e64 v0, s8, s31                                 // 00000000185C: D52B0000 00003E08
	v_fmac_f32_e64 v0, s9, s16                                 // 000000001864: D52B0000 00002009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000186C: BF870091
	v_fmac_f32_e64 v0, s10, s17                                // 000000001870: D52B0000 0000220A
	v_fmac_f32_e64 v0, s11, s18                                // 000000001878: D52B0000 0000240B
	s_waitcnt lgkmcnt(0)                                       // 000000001880: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001884: BF8700C1
	v_fmac_f32_e64 v0, s15, s19                                // 000000001888: D52B0000 0000260F
	s_load_b32 s15, s[12:13], 0x278                            // 000000001890: F40003C6 F8000278
	s_load_b128 s[8:11], s[36:37], 0x60                        // 000000001898: F4080212 F8000060
	s_load_b128 s[16:19], s[12:13], 0x284                      // 0000000018A0: F4080406 F8000284
	v_fmac_f32_e64 v0, s0, s20                                 // 0000000018A8: D52B0000 00002800
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018B0: BF8700A1
	v_fmac_f32_e64 v0, s1, s21                                 // 0000000018B4: D52B0000 00002A01
	s_load_b64 s[0:1], s[36:37], 0x70                          // 0000000018BC: F4040012 F8000070
	v_fmac_f32_e64 v0, s2, s22                                 // 0000000018C4: D52B0000 00002C02
	s_load_b32 s2, s[12:13], 0x294                             // 0000000018CC: F4000086 F8000294
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000018D4: BF8700B1
	v_fmac_f32_e64 v0, s3, s23                                 // 0000000018D8: D52B0000 00002E03
	s_add_u32 s3, s4, s34                                      // 0000000018E0: 80032204
	s_waitcnt lgkmcnt(0)                                       // 0000000018E4: BF89FC07
	v_fmac_f32_e64 v0, s15, s8                                 // 0000000018E8: D52B0000 0000100F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018F0: BF870091
	v_fmac_f32_e64 v0, s16, s9                                 // 0000000018F4: D52B0000 00001210
	v_fmac_f32_e64 v0, s17, s10                                // 0000000018FC: D52B0000 00001411
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001904: BF870091
	v_fmac_f32_e64 v0, s18, s11                                // 000000001908: D52B0000 00001612
	v_fmac_f32_e64 v0, s19, s0                                 // 000000001910: D52B0000 00000013
	s_mul_i32 s0, s14, 3                                       // 000000001918: 9600830E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000191C: BF870001
	v_fmac_f32_e64 v0, s2, s1                                  // 000000001920: D52B0000 00000202
	s_addc_u32 s2, s5, s35                                     // 000000001928: 82022305
	s_ashr_i32 s1, s0, 31                                      // 00000000192C: 86019F00
	v_mov_b32_e32 v1, 0                                        // 000000001930: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001934: 84808200
	v_max_f32_e32 v0, 0, v0                                    // 000000001938: 20000080
	s_add_u32 s0, s3, s0                                       // 00000000193C: 80000003
	s_addc_u32 s1, s2, s1                                      // 000000001940: 82010102
	s_add_u32 s0, s0, s6                                       // 000000001944: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001948: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 00000000194C: DC6A0000 00000001
	s_nop 0                                                    // 000000001954: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001958: BFB60003
	s_endpgm                                                   // 00000000195C: BFB00000
