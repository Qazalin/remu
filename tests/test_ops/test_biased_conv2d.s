
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_8_25_8>:
	s_load_b256 s[0:7], s[0:1], null                           // 000000001700: F40C0000 F8000000
	s_mov_b32 s12, s15                                         // 000000001708: BE8C000F
	s_ashr_i32 s13, s15, 31                                    // 00000000170C: 860D9F0F
	v_mov_b32_e32 v1, 0                                        // 000000001710: 7E020280
	s_lshl_b64 s[8:9], s[12:13], 2                             // 000000001714: 8488820C
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s16, s6, s8                                      // 00000000171C: 80100806
	s_addc_u32 s17, s7, s9                                     // 000000001720: 82110907
	s_ashr_i32 s15, s14, 31                                    // 000000001724: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001728: BF870499
	s_lshl_b64 s[14:15], s[14:15], 2                           // 00000000172C: 848E820E
	s_add_u32 s2, s2, s14                                      // 000000001730: 80020E02
	s_addc_u32 s3, s3, s15                                     // 000000001734: 82030F03
	s_lshl_b32 s6, s12, 3                                      // 000000001738: 8406830C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_ashr_i32 s7, s6, 31                                      // 000000001740: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001744: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001748: BF870009
	s_add_u32 s4, s4, s6                                       // 00000000174C: 80040604
	s_addc_u32 s5, s5, s7                                      // 000000001750: 82050705
	s_load_b256 s[4:11], s[4:5], null                          // 000000001754: F40C0102 F8000000
	s_clause 0x7                                               // 00000000175C: BF850007
	s_load_b32 s13, s[2:3], null                               // 000000001760: F4000341 F8000000
	s_load_b32 s18, s[2:3], 0x64                               // 000000001768: F4000481 F8000064
	s_load_b32 s19, s[2:3], 0xc8                               // 000000001770: F40004C1 F80000C8
	s_load_b32 s20, s[2:3], 0x12c                              // 000000001778: F4000501 F800012C
	s_load_b32 s21, s[2:3], 0x190                              // 000000001780: F4000541 F8000190
	s_load_b32 s22, s[2:3], 0x1f4                              // 000000001788: F4000581 F80001F4
	s_load_b32 s23, s[2:3], 0x258                              // 000000001790: F40005C1 F8000258
	s_load_b32 s2, s[2:3], 0x2bc                               // 000000001798: F4000081 F80002BC
	s_load_b32 s3, s[16:17], null                              // 0000000017A0: F40000C8 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000017A8: BF89FC07
	v_fma_f32 v0, s13, s4, 0                                   // 0000000017AC: D6130000 0200080D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B4: BF870091
	v_fmac_f32_e64 v0, s18, s5                                 // 0000000017B8: D52B0000 00000A12
	v_fmac_f32_e64 v0, s19, s6                                 // 0000000017C0: D52B0000 00000C13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C8: BF870091
	v_fmac_f32_e64 v0, s20, s7                                 // 0000000017CC: D52B0000 00000E14
	v_fmac_f32_e64 v0, s21, s8                                 // 0000000017D4: D52B0000 00001015
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017DC: BF870091
	v_fmac_f32_e64 v0, s22, s9                                 // 0000000017E0: D52B0000 00001216
	v_fmac_f32_e64 v0, s23, s10                                // 0000000017E8: D52B0000 00001417
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017F0: BF8700A1
	v_fmac_f32_e64 v0, s2, s11                                 // 0000000017F4: D52B0000 00001602
	s_mul_i32 s2, s12, 25                                      // 0000000017FC: 9602990C
	v_add_f32_e32 v0, s3, v0                                   // 000000001800: 06000003
	s_ashr_i32 s3, s2, 31                                      // 000000001804: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001808: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000180C: 84828202
	v_max_f32_e32 v0, 0, v0                                    // 000000001810: 20000080
	s_add_u32 s0, s0, s2                                       // 000000001814: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001818: 82010301
	s_add_u32 s0, s0, s14                                      // 00000000181C: 80000E00
	s_addc_u32 s1, s1, s15                                     // 000000001820: 82010F01
	global_store_b32 v1, v0, s[0:1]                            // 000000001824: DC6A0000 00000001
	s_nop 0                                                    // 00000000182C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001830: BFB60003
	s_endpgm                                                   // 000000001834: BFB00000
