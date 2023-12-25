
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_10_5_2_3>:
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
	s_mul_i32 s2, s15, 6                                       // 00000000173C: 9602860F
	s_add_u32 s8, s8, s6                                       // 000000001740: 80080608
	s_addc_u32 s9, s9, s7                                      // 000000001744: 82090709
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_add_u32 s10, s0, s2                                      // 000000001754: 800A0200
	s_addc_u32 s11, s1, s3                                     // 000000001758: 820B0301
	s_load_b128 s[0:3], s[10:11], null                         // 00000000175C: F4080005 F8000000
	s_clause 0x3                                               // 000000001764: BF850003
	s_load_b64 s[12:13], s[8:9], null                          // 000000001768: F4040304 F8000000
	s_load_b32 s18, s[8:9], 0x8                                // 000000001770: F4000484 F8000008
	s_load_b64 s[16:17], s[8:9], 0x1c                          // 000000001778: F4040404 F800001C
	s_load_b32 s19, s[8:9], 0x24                               // 000000001780: F40004C4 F8000024
	s_load_b64 s[8:9], s[10:11], 0x10                          // 000000001788: F4040205 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001790: BF89FC07
	v_fma_f32 v0, s12, s0, 0                                   // 000000001794: D6130000 0200000C
	s_mul_i32 s0, s15, 50                                      // 00000000179C: 9600B20F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017A0: BF8704A1
	v_fmac_f32_e64 v0, s13, s1                                 // 0000000017A4: D52B0000 0000020D
	s_ashr_i32 s1, s0, 31                                      // 0000000017AC: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017B0: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017B4: BF8700C1
	v_fmac_f32_e64 v0, s18, s2                                 // 0000000017B8: D52B0000 00000412
	s_mul_i32 s2, s14, 5                                       // 0000000017C0: 9602850E
	s_add_u32 s4, s4, s0                                       // 0000000017C4: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017C8: 82050105
	v_fmac_f32_e64 v0, s16, s3                                 // 0000000017CC: D52B0000 00000610
	s_ashr_i32 s3, s2, 31                                      // 0000000017D4: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000017DC: 84808202
	v_fmac_f32_e64 v0, s17, s8                                 // 0000000017E0: D52B0000 00001011
	s_add_u32 s0, s4, s0                                       // 0000000017E8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017EC: 82010105
	s_add_u32 s0, s0, s6                                       // 0000000017F0: 80000600
	s_addc_u32 s1, s1, s7                                      // 0000000017F4: 82010701
	v_fmac_f32_e64 v0, s19, s9                                 // 0000000017F8: D52B0000 00001213
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001800: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001804: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 00000000180C: DC6A0000 00000001
	s_nop 0                                                    // 000000001814: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001818: BFB60003
	s_endpgm                                                   // 00000000181C: BFB00000
